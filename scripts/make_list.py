#!/usr/bin/env python3
"""把 data/train_manifest.json 变成 GPT-SoVITS 训练用的 .list 文件。

GPT-SoVITS 的微调入口认的是一份纯文本清单，每行：

    <wav 路径>|<说话人>|<语言>|<文本>

注意这四段用 | 分隔，所以文本里本身不能含 |（GPT-SoVITS 内部是 line.split("|")，
多一个 | 会直接被 try/except 吃掉，表现为这条数据静默消失）。

输出写到 logs/<exp_name>/dataset.list —— 微调脚本默认在那个目录里找它。
同时把 wav 复制到 logs/<exp_name>/wavs/，让清单里的路径在 CI 上稳定可复现
（不依赖 data/ 这个目录名，也不受后续步骤改名影响）。
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data" / "train_manifest.json"
    if not manifest_path.exists():
        print(f"找不到清单文件: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    speaker = manifest.get("speaker", "demo")
    language = manifest.get("language", "zh")
    clips = manifest.get("clips", [])

    exp_name = os.environ.get("EXP_NAME", "demo_voice")
    out_dir = root / "logs" / exp_name
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    missing = []
    for clip in clips:
        src = root / "data" / "raw" / "train" / clip["wav"]
        if not src.exists():
            missing.append(str(src))
            continue
        dst = wav_dir / clip["wav"]
        shutil.copy2(src, dst)
        text = clip["text"].replace("|", "，").replace("\n", " ").strip()
        if not text:
            missing.append(f"{src}（文本为空）")
            continue
        lines.append(f"{dst.resolve()}|{speaker}|{language}|{text}")

    if missing:
        print("以下音频缺失或无效，已跳过：", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)

    if not lines:
        print("清单里一条可用数据都没有，无法训练", file=sys.stderr)
        return 1

    list_path = out_dir / "dataset.list"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf8")

    # 汇总总时长，方便判断这批数据够不够（经验上 <5 分钟效果会明显变差）
    total = 0.0
    for clip in clips:
        src = root / "data" / "raw" / "train" / clip["wav"]
        if not src.exists():
            continue
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True,
        )
        try:
            total += float(probe.stdout.strip())
        except ValueError:
            pass

    print(f"写入 {list_path}，共 {len(lines)} 条，总时长 {total:.1f}s")
    if total < 60:
        print("  [提醒] 训练集不到 1 分钟，微调结果会比较像原音色但口音/情绪提升有限；"
              "想更好就准备 5-10 分钟。")
    for line in lines:
        print("  " + line.replace(str(root), "<repo>"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
