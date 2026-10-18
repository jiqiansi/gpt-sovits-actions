#!/usr/bin/env bash
# 在 GitHub Actions runner（或任意 Ubuntu 24.04 CPU 机器）上装好能跑 GPT-SoVITS 的 Python 环境。
#
# 设计要点：
#   1. 不用 conda —— runner 上现成的是 Miniconda/base(3.12)，而 GPT-SoVITS 要求 3.10。
#      直接用 python3.10 -m venv，省掉 conda 的 solver 和几 GB 磁盘。
#   2. torch 走 CPU 专用 wheel。默认 PyPI 的 torch 是 CUDA 版（2GB+），
#      CPU 版只有 ~174MB。这里从 download.pytorch.org/whl/cpu 装。
#   3. 所有大下载（pip / huggingface）都套 5 次重试 + 长超时。
#      CI 上单次 TLS 断流很常见，不重试就会红。
#   4. requirements.txt 里三个 x86_64 上会出问题的包被剔除：
#        onnxruntime-gpu  → 换成 CPU 版 onnxruntime
#        torch / torchaudio → 已在前面按 CPU wheel 装好
#        --no-binary=opencc  → 会在 CI 上硬编译，改吃现成 wheel
#   5. pyopenjtalk / jieba_fast 没有 cp310 wheel，必须现场编译。
#      需要 cmake + gcc-11（gcc-13 编 pyopenjtalk 会挂）。见 apt 那一段。
set -uo pipefail

PY=${PY:-python3.10}
REPO_DIR=${REPO_DIR:-$PWD}
GSV_DIR="$REPO_DIR/GPT-SoVITS"
mark() { echo; echo "########## $* ##########"; }

mark "运行环境"
echo "runner: $(uname -a)"
nproc | sed 's/^/nproc: /'
df -h / | tail -1
$PY -V

# 缓存目录：GitHub runner 的 /mnt 属 root，普通用户 mkdir 会静默失败，
# 结果就是后面重定向到那里直接 ENOENT。所以默认放在 workspace 里（已在 .gitignore）。
CACHE=${CACHE:-"$REPO_DIR/.cache"}
mkdir -p "$CACHE" || { echo "!! 建不了缓存目录 $CACHE"; exit 1; }
echo "cache dir: $CACHE ($(df -h "$CACHE" | tail -1 | awk '{print $4}') free)"

mark "APT: cmake / gcc-11 / ffmpeg"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
# gymnasium 之类会拖住 apt，加 --no-install-recommends 快速通过
sudo apt-get install -y -qq --no-install-recommends \
  cmake ninja-build build-essential gcc-11 g++-11 ffmpeg unzip libsndfile1 curl git > /dev/null
# pyopenjtalk 用 gcc-13 编译会因为 C++ 标准库头文件报错，钉死 gcc-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
gcc --version | head -1
cmake --version | head -1
ffmpeg -version | head -1

mark "venv"
if [ ! -x "$REPO_DIR/.venv/bin/python" ]; then
  $PY -m venv "$REPO_DIR/.venv"
fi
PIP=("$REPO_DIR/.venv/bin/python" -m pip)
"${PIP[@]}" install -q --upgrade pip wheel setuptools
VENV_PY="$REPO_DIR/.venv/bin/python"
$VENV_PY -V

# 通用重试封装：CI 上网络抖动是常态，别让一次断流毁掉整个 job
retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    if [ "$n" -ge 5 ]; then
      echo "!! 重试 5 次仍失败: $*"
      return 1
    fi
    echo "-- 第 $n 次失败，10s 后重试: $*"
    sleep 10
  done
}

mark "PyTorch (CPU wheel)"
# CPU-only 索引。默认 PyPI 的 torch 是 CUDA 版，在无 GPU runner 上白下 2GB。
retry "${PIP[@]}" install --no-cache-dir \
  torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
$VENV_PY - <<'EOF'
import torch, torchaudio
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("torchaudio", torchaudio.__version__)
assert not torch.cuda.is_available(), "runner 上不该有 CUDA"
EOF

mark "GPT-SoVITS requirements.txt"
REQ="$CACHE/reqs-fixed.txt"
if [ ! -f "$REQ" ] || [ "${REFRESH_REQS:-0}" = "1" ]; then
  # 过滤掉三条在 x86_64 CI 上会出问题的行（原因见文件头注释）
  grep -v -E '^(torch$|torchaudio$|onnxruntime-gpu|onnxruntime;|--no-binary)' \
    "$GSV_DIR/requirements.txt" > "$REQ"
  # soundfile：webui/api 保存 wav 时用，requirements 里没有（靠 librosa 间接带进来）
  echo "soundfile" >> "$REQ"
  # CPU 版 onnxruntime，替代 requirements 里 x86_64 才会装的 onnxruntime-gpu
  echo "onnxruntime" >> "$REQ"
fi
echo "--- 实际安装的依赖（已过滤）---"
cat "$REQ"
retry "${PIP[@]}" install --no-cache-dir -r "$REQ"

mark "校验关键 import"
# 这一段是整条流水线的闸门。它必须真的能把 job 拦下来 ——
# 之前这里失败了却继续往下跑，后面的步骤在空 venv 里连 numpy 都没有，
# 报出来的错离根因十万八千里。所以显式检查退出码，别指望 set -e。
if ! $VENV_PY - <<'EOF'
import importlib, sys
mods = ["torch", "torchaudio", "transformers", "librosa", "peft",
        "pytorch_lightning", "onnxruntime", "pyopenjtalk",
        "jieba_fast", "soundfile", "numpy", "scipy", "fast_langdetect",
        "split_lang", "wordsegment", "g2p_en", "cn2an", "pypinyin", "opencc"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append((m, repr(e)[:160]))
for m, e in bad:
    print("FAIL", m, e)
print("OK" if not bad else f"{len(bad)} 个模块导入失败")
sys.exit(1 if bad else 0)
EOF
then
  echo "!! 依赖没装全，后面每一步都会莫名其妙失败，这里直接停。"
  echo "   已装的包："
  "$VENV_PY" -m pip list 2>/dev/null | head -40
  exit 1
fi

mark "bootstrap 完成"
echo "venv: $REPO_DIR/.venv"
du -sh "$REPO_DIR/.venv"
# 磁盘吃紧时清掉 pip 缓存，后面还要下 1.6GB 权重
"$VENV_PY" -m pip cache purge >/dev/null 2>&1 || true
df -h / | tail -1
