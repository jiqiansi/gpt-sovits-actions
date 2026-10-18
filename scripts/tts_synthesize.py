#!/usr/bin/env python3
"""GPT-SoVITS 推理：参考音频 + 目标文本 -> 合成 wav。

不走 api_v2.py（那是常驻服务），直接调 TTS_infer_pack.TTS，一个进程出一个文件。

关于配置：TTS_Config 收一个 dict，字段和 configs/tts_infer.yaml 的 v2 段一样。
CPU 上必须 device="cpu" + is_half=False；is_half=True 在 CPU 上要么报错要么极慢。

权重来源优先级（由 --gpt/--sovits 决定，默认自动挑最新微调产物）：
  1. 微调产物 GPT_weights_v2/ 与 SoVITS_weights_v2/（scripts/train.sh 产出）
  2. 预训练权重（GPT-SoVITS/pretrained_models/...）
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_gsv_root() -> Path:
    """定位 GPT-SoVITS 的仓库根目录。

    上游把所有相对路径都写成 `GPT_SoVITS/pretrained_models/...`、`tools/...`，
    也就是说必须以 GPT-SoVITS 那个目录当 cwd 才能跑。
    CI 上它被 clone 到 <repo>/GPT-SoVITS/，本地开发时可能就是 <repo> 本身。
    """
    for cand in (ROOT / "GPT-SoVITS", ROOT):
        if (cand / "GPT_SoVITS" / "TTS_infer_pack").is_dir() and (cand / "tools").is_dir():
            return cand
    raise SystemExit(f"找不到 GPT-SoVITS 目录，试过 {ROOT / 'GPT-SoVITS'} 和 {ROOT}")


GSV = find_gsv_root()
# TTS_infer_pack 内部有大量 `from tools.xxx import` / `from module.xxx import`，
# 这些包位于 GPT_SoVITS/ 下，所以 GPT_SoVITS 目录本身必须在 sys.path 里。
sys.path.insert(0, str(GSV))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))
os.chdir(GSV)   # 上游所有权重/模型路径都是相对这个 cwd 写的


def pick_latest(pattern_dir: Path, suffix: str):
    """挑出目录里最新的一个权重文件（文件名里 epoch/step 数字最大的那个）。"""
    if not pattern_dir.is_dir():
        return None
    cands = [p for p in pattern_dir.glob(f"*{suffix}") if p.is_file()]
    if not cands:
        return None

    def key(p: Path):
        nums = [int(t) for t in __import__("re").findall(r"\d+", p.stem)]
        return (nums or [0], p.stat().st_mtime)

    return max(cands, key=key)


def default_weights():
    gpt = pick_latest(ROOT / "GPT_weights_v2", ".ckpt")
    sov = pick_latest(ROOT / "SoVITS_weights_v2", ".pth")
    return (
        str(gpt) if gpt else "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        str(sov) if sov else "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    )


def resolve(p: str) -> Path:
    """把用户给的相对路径落到实际存在的目录。

    cwd 已经切到 GSV（为了兼容上游那堆相对路径），但 --out / --ref-audio
    这类由本仓库提供的路径是相对仓库根写死的，两个地方都要试。
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    for base in (ROOT, GSV):
        if (base / pp).exists():
            return base / pp
    return ROOT / pp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="要合成的目标文本")
    ap.add_argument("--lang", default="zh", help="目标文本语言，zh/en/ja/ko/yue")
    ap.add_argument("--ref-audio", default="data/raw/ref.wav", help="参考音频（3-10 秒干净人声）")
    ap.add_argument("--ref-text", default=None, help="参考音频对应的文字；不给就用 data/ref_text.txt")
    ap.add_argument("--gpt", default=None, help="GPT 权重 .ckpt 路径")
    ap.add_argument("--sovits", default=None, help="SoVITS 权重 .pth 路径")
    ap.add_argument("--out", default="data/out/tts_out.wav")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    ref_audio = resolve(args.ref_audio)
    if not ref_audio.exists():
        print(f"参考音频不存在: {ref_audio}", file=sys.stderr)
        return 1

    if args.ref_text is None:
        ref_text_file = resolve("data/ref_text.txt")
        if ref_text_file.exists():
            args.ref_text = ref_text_file.read_text(encoding="utf8").strip()
        else:
            print("没给 --ref-text，也没有 data/ref_text.txt", file=sys.stderr)
            return 1

    gpt_path, sovits_path = args.gpt, args.sovits
    if not gpt_path or not sovits_path:
        dg, ds = default_weights()
        gpt_path = gpt_path or dg
        sovits_path = sovits_path or ds
    for p in (gpt_path, sovits_path):
        if not resolve(p).exists():
            print(f"权重不存在: {p}（cwd={Path.cwd()}）", file=sys.stderr)
            return 1
    gpt_path, sovits_path = str(resolve(gpt_path)), str(resolve(sovits_path))

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("GPT    :", gpt_path)
    print("SoVITS :", sovits_path)
    print("ref    :", ref_audio)

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    cfg = {
        "device": "cpu",
        "is_half": False,
        "version": "v2",
        "t2s_weights_path": gpt_path,
        "vits_weights_path": sovits_path,
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    }
    t0 = time.time()
    tts = TTS(TTS_Config(cfg))
    print(f"[加载模型] {time.time() - t0:.1f}s")

    t0 = time.time()
    # run() 是个生成器，逐段吐 (采样率, int16 波形)；必须全量收集，否则音频会被截断
    inputs = {
        "text": args.text,
        "text_lang": args.lang,
        "ref_audio_path": str(ref_audio),
        "aux_ref_audio_paths": [],
        "prompt_text": args.ref_text,
        "prompt_lang": args.lang,
        "top_k": args.top_k,
        "top_p": 1,
        "temperature": args.temperature,
        "text_split_method": "cut5",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": args.seed,
        "parallel_infer": False,   # CPU 上多线程抢核，关掉更稳
        "repetition_penalty": 1.35,
    }
    sr, audio = None, None
    chunks = []
    for item in tts.run(inputs):
        # 部分版本返回 (sr, audio)，部分返回 (sr, audio, ...)；统一按前两个取
        sr_i, audio_i = item[0], item[1]
        chunks.append(audio_i)
        sr = sr_i
    import numpy as np

    if not chunks:
        print("TTS 没有产出任何音频", file=sys.stderr)
        return 1
    audio = np.concatenate([np.asarray(c).reshape(-1) for c in chunks], axis=0)
    dur = len(audio) / sr
    print(f"[推理] {time.time() - t0:.1f}s，输出 {dur:.2f}s @ {sr}Hz")

    try:
        import soundfile as sf

        sf.write(str(out_path), audio, sr)
    except Exception as exc:
        # soundfile 缺失或写不动时退回 scipy
        print(f"[警告] soundfile 写入失败({exc!r})，改用 scipy", file=sys.stderr)
        from scipy.io import wavfile

        wavfile.write(str(out_path), sr, audio.astype("int16"))

    info = {"gpt": gpt_path, "sovits": sovits_path, "sr": int(sr),
            "duration_sec": round(float(dur), 2), "text": args.text,
            "ref_audio": str(ref_audio), "gsv_root": str(GSV)}
    out_path.with_suffix(out_path.suffix + ".json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print("写出:", out_path, out_path.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
