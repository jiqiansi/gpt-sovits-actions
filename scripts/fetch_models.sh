#!/usr/bin/env bash
# 按需下载 GPT-SoVITS 预训练权重（v2，CPU 训练 + 推理的最小集合）。
#
# 上游 install.sh 会下一个 4.56GB 的 pretrained_models.zip，里面绝大部分是我们用不到的
# （v3/v4/v5/v2Pro 的权重、BigVGAN、uVR5、faster-whisper、funasr……）。
# 这里只下 7 个文件，合计约 1.6GB，比 install.sh 少 2.9GB —— 这直接决定
# 能不能塞进 GitHub runner 的磁盘预算里。
#
# 下载地址可以整体切换（国内网络建议 HF_MIRROR=1）：
#   HF_BASE=https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main
#   HF_BASE=https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main
set -uo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
GSV="$REPO_DIR/GPT-SoVITS"
PM="$GSV/GPT_SoVITS/pretrained_models"
G2PW_DIR="$GSV/GPT_SoVITS/text/G2PWModel"

if [ "${HF_MIRROR:-0}" = "1" ]; then
  HF_BASE=${HF_BASE:-https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main}
else
  HF_BASE=${HF_BASE:-https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main}
fi

mark() { echo; echo "########## $* ##########"; }
mark "权重下载（base = $HF_BASE）"

# fetch <远程相对路径> <本地绝对路径>
# 已存在且非空就跳过，这样 Actions 缓存命中时这一步是秒过的。
fetch() {
  local remote="$1" local_path="$2"
  if [ -s "$local_path" ]; then
    echo "  [skip] $local_path ($(stat -c%s "$local_path") bytes)"
    return 0
  fi
  mkdir -p "$(dirname "$local_path")"
  local tmp="$local_path.part"
  local n=0
  while [ $n -lt 5 ]; do
    echo "  [get ] $remote"
    # -L 跟随 302；-C - 断点续传；--retry 交给 curl 内部再兜一层
    if curl -L -C - --retry 3 --retry-delay 5 --connect-timeout 30 \
         --max-time 1800 -o "$tmp" "$HF_BASE/$remote"; then
      mv "$tmp" "$local_path"
      echo "        -> $(stat -c%s "$local_path") bytes"
      return 0
    fi
    n=$((n + 1))
    echo "        第 $n 次失败，15s 后重试"
    sleep 15
  done
  echo "!! 下载失败: $remote" >&2
  return 1
}

# ---------------------------------------------------------------- 推理 + 训练都要
# 中文 BERT（音素级特征，v2 必需）
fetch "pretrained_models/chinese-roberta-wwm-ext-large/config.json"        "$PM/chinese-roberta-wwm-ext-large/config.json"
fetch "pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json"     "$PM/chinese-roberta-wwm-ext-large/tokenizer.json"
fetch "pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin"  "$PM/chinese-roberta-wwm-ext-large/pytorch_model.bin"
# 中文 HuBERT（内容特征）
fetch "pretrained_models/chinese-hubert-base/config.json"                  "$PM/chinese-hubert-base/config.json"
fetch "pretrained_models/chinese-hubert-base/preprocessor_config.json"     "$PM/chinese-hubert-base/preprocessor_config.json"
fetch "pretrained_models/chinese-hubert-base/pytorch_model.bin"            "$PM/chinese-hubert-base/pytorch_model.bin"
# v2 预训练 GPT(t2s) 与 SoVITS 生成器
fetch "pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" \
      "$PM/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
fetch "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"              "$PM/gsv-v2final-pretrained/s2G2333k.pth"
# 语种自动识别（text_lang=auto 时用，LangSegmenter 会在 import 时就找它）
fetch "pretrained_models/fast_langdetect/lid.176.ftz"                      "$PM/fast_langdetect/lid.176.ftz"

# ------------------------------------------------------------------ 训练专用
# s2_train.py 需要判别器做微调；纯推理不需要
if [ "${NEED_D:-1}" = "1" ]; then
  fetch "pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"            "$PM/gsv-v2final-pretrained/s2D2333k.pth"
else
  echo "  [skip] s2D2333k.pth（NEED_D=0，纯推理不需要判别器）"
fi

# ------------------------------------------------- G2PW 多音字模型（中文必装）
# text/chinese2.py 在模块加载时就 new 一个 G2PWPinyin(model_dir="GPT_SoVITS/text/G2PWModel")，
# 目录不存在它会自己去 modelscope 拉一个 589MB 的包 —— 那条路径在 CI 上很慢且不透明，
# 所以这里直接把 zip 拉下来解到它期望的位置。
if [ ! -f "$G2PW_DIR/g2pW.onnx" ]; then
  mark "G2PW 多音字模型"
  TMPZ=$(mktemp /tmp/g2pw.XXXXXX.zip)
  ok=0
  for i in 1 2 3 4 5; do
    if curl -L --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 1800 \
         -o "$TMPZ" "$HF_BASE/G2PWModel.zip"; then ok=1; break; fi
    echo "  第 $i 次失败，15s 后重试"; sleep 15
  done
  [ "$ok" = "1" ] || { echo "!! G2PWModel.zip 下载失败" >&2; rm -f "$TMPZ"; exit 1; }
  echo "  zip: $(stat -c%s "$TMPZ") bytes"
  unzip -o -q "$TMPZ" -d "$GSV/GPT_SoVITS/text/"
  rm -f "$TMPZ"
else
  echo "  [skip] G2PWModel 已存在"
fi

# ------------------------------------------------- nltk 数据（英文/混合语句要用）
# 中文单独跑其实用不到，但目标文本里只要出现拉丁字母（"GitHub Actions"、"CPU"），
# LangSegmenter 就会切出一个英文段并走 english.py -> nltk.pos_tag，
# 缺数据直接 LookupError: averaged_perceptron_tagger_eng。
#
# 位置是硬要求：nltk.data.path 只查 NLTK_DATA / ~/nltk_data / sys.prefix/nltk_data
# 等少数几处，**没有 CWD 相对路径**。所以必须解到 venv 根下（= sys.prefix/nltk_data），
# 这也正是上游 install.sh 的 `unzip -d "$PY_PREFIX"` 在做的事。
# 之前解到 GPT-SoVITS/nltk_data 是解了个寂寞 —— 文件都在，就是没人去找。
mark "nltk 数据"
NLTK_DIR="${VENV_PREFIX:-$REPO_DIR/.venv}/nltk_data"
if [ ! -d "$NLTK_DIR/taggers" ]; then
  mkdir -p "$(dirname "$NLTK_DIR")"
  TMPN=$(mktemp /tmp/nltk.XXXXXX.zip)
  ok=0
  for i in 1 2 3 4 5; do
    if curl -L --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 600 \
         -o "$TMPN" "$HF_BASE/nltk_data.zip"; then ok=1; break; fi
    echo "  第 $i 次失败，10s 后重试"; sleep 10
  done
  if [ "$ok" = "1" ]; then
    # zip 顶层就是 nltk_data/ 目录，解到 venv 根下正好生成 <venv>/nltk_data
    unzip -o -q "$TMPN" -d "$(dirname "$NLTK_DIR")"
    rm -f "$TMPN"
  else
    echo "  [warn] nltk_data 下载失败，含英文的文本会不可用（纯中文不受影响）"
  fi
else
  echo "  [skip] nltk_data 已存在 ($NLTK_DIR)"
fi
echo "  tagger: $(ls "$NLTK_DIR/taggers" 2>/dev/null | tr '\n' ' ')"

mark "权重校验"
echo "--- 文件清单与大小 ---"
find "$PM" "$G2PW_DIR" -type f -printf '%10s  %p\n' 2>/dev/null | sort -k2
echo "--- 总占用 ---"
du -sh "$PM" "$G2PW_DIR" 2>/dev/null

mark "关键文件存在性检查"
fail=0
for f in \
  "$PM/chinese-roberta-wwm-ext-large/pytorch_model.bin" \
  "$PM/chinese-hubert-base/pytorch_model.bin" \
  "$PM/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" \
  "$PM/gsv-v2final-pretrained/s2G2333k.pth" \
  "$G2PW_DIR/g2pW.onnx"; do
  if [ -s "$f" ]; then
    echo "  ok   $(basename "$f")  $(stat -c%s "$f") bytes"
  else
    echo "  FAIL $(basename "$f")"; fail=1
  fi
done
[ "$fail" = "1" ] && { echo "!! 关键权重缺失"; exit 1; }
echo "全部就绪"
