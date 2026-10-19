#!/usr/bin/env bash
# GPT-SoVITS 微调全链路（CPU 版）。
#
# 一次跑完 WebUI 上的四个按钮：
#   ① 语音切分（这里跳过，因为 data/raw/train/ 里已经是切好的短句）
#   ② 开源格式 -> 2-name2text.txt（BERT 音素特征）
#   ③ -> 4-cnhubert/*.pt + 5-wav32k/*.wav（HuBERT 特征与 32k 音频）
#   ④ -> 6-name2semantic.tsv（用预训练 SoVITS 提取语义 token）
#   ⑤ s1_train.py 微调 GPT（t2s 模型）
#   ⑥ s2_train.py 微调 SoVITS
#
# 全部靠环境变量驱动，和 WebUI 的 open1a/open1Bb/open1Ba 完全一致。
#
# CPU 上的两处关键调整（不改必出问题）：
#   is_half=False  —— 半精度只在 CUDA 上有意义，CPU 上强行 .half() 会变慢甚至 NaN
#   s1 precision=32 —— 3.16-mixed 在 CPU 上会被 pytorch-lightning 静默换成 bf16-mixed，
#                      既慢又可能丢精度，直接钉成 32
set -uo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
GSV="$REPO_DIR/GPT-SoVITS"
VENV_PY=${VENV_PY:-"$REPO_DIR/.venv/bin/python"}

EXP_NAME=${EXP_NAME:-demo_voice}
VERSION=${VERSION:-v2}
GPT_EPOCHS=${GPT_EPOCHS:-12}
S2_EPOCHS=${S2_EPOCHS:-8}
GPT_BS=${GPT_BS:-2}
S2_BS=${S2_BS:-2}

LIST_FILE="$REPO_DIR/logs/$EXP_NAME/dataset.list"
OUT="$REPO_DIR/logs/$EXP_NAME"

mark() { echo; echo "########## $* ##########"; }

# 预训练权重：版本不同文件名不同，统一收敛到两个变量里
case "$VERSION" in
  v1)
    PRE_S1="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    PRE_S2G="GPT_SoVITS/pretrained_models/s2G488k.pth"
    PRE_S2D="GPT_SoVITS/pretrained_models/s2D488k.pth"
    ;;
  v2|*)
    PRE_S1="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    PRE_S2G="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    PRE_S2D="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
    ;;
esac
# 配置在 GPT_SoVITS/configs/ 下（仓库根没有 configs/ 这一层）
S2_CONFIG="$GSV/GPT_SoVITS/configs/s2.json"
S1_TPL="$GSV/GPT_SoVITS/configs/s1longer-v2.yaml"

cd "$GSV" || exit 1
# 权重输出目录必须写成绝对路径。
# 上游 webui 用的是相对路径 "GPT_weights_v2"，而它在 GPT-SoVITS 目录里启动，
# 所以权重落在那里。我们这里的 cwd 同样是 $GSV，但 workflow 收集产物和
# tts_synthesize.py 找最新权重都是按 workspace 根算的 —— 写相对路径的话
# 训练会成功、产物却永远收集不到，而且两边都不会报错。
# 上游按版本把导出目录分开放：v1 是 GPT_weights / SoVITS_weights，
# v2 及以后才带 _v2 后缀。这里跟 config.py 里的 GPT_weight_version2root 对齐。
GPT_W="$REPO_DIR/GPT_weights$([ "$VERSION" = v1 ] && echo "" || echo "_v2")"
SOV_W="$REPO_DIR/SoVITS_weights$([ "$VERSION" = v1 ] && echo "" || echo "_v2")"
mkdir -p "$GPT_W" "$SOV_W"
export GPT_W SOV_W

export PYTHONPATH="$GSV:$GSV/GPT_SoVITS:${PYTHONPATH:-}"
export is_half=False
export version="$VERSION"
export hz=25hz
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$(nproc)
# nltk 在 CI 上只找 NLTK_DATA / sys.prefix/nltk_data，显式指一份更保险
[ -d "$REPO_DIR/.venv/nltk_data" ] && export NLTK_DATA="$REPO_DIR/.venv/nltk_data"

# 三个 prepare 脚本共用的环境变量
export inp_text="$LIST_FILE"
export inp_wav_dir=""
export exp_name="$EXP_NAME"
export i_part=0
export all_parts=1
export opt_dir="$OUT"

run_py() {
  local name="$1"; shift
  mark "$name"
  echo "python $*"
  "$VENV_PY" "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo "!! $name 失败 (exit $rc)"; exit $rc; }
  return 0
}

[ -f "$LIST_FILE" ] || { echo "找不到 $LIST_FILE，先跑 scripts/make_list.py"; exit 1; }
mkdir -p "$OUT"
# 两个配置模板先验：路径写错时 utils.get_hparams 会在很后面才报 FileNotFoundError，
# 而那时前面三个 prepare 脚本已经跑完，白等几分钟还不一定看得出根因。
for cfg in "$S1_TPL" "$S2_CONFIG"; do
  [ -f "$cfg" ] || { echo "!! 配置不存在: $cfg（应该在 GPT_SoVITS/configs/ 下）" >&2; exit 1; }
done
echo "s1 模板: $S1_TPL"
echo "s2 模板: $S2_CONFIG"

# ---------------------------------------------------------------- ② 音素 + BERT
mark "② 1-get-text.py：生成 2-name2text.txt（音素序列 + 中文 BERT 特征）"
export bert_pretrained_dir="$GSV/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
run_py "1-get-text" GPT_SoVITS/prepare_datasets/1-get-text.py
# WebUI 这里会把 2-name2text-<i>.txt 合并成 2-name2text.txt；单进程下就是重命名
cat "$OUT/2-name2text-0.txt" > "$OUT/2-name2text.txt"
echo "2-name2text.txt 行数: $(wc -l < "$OUT/2-name2text.txt")"

# ------------------------------------------------------------ ③ HuBERT + 32k wav
mark "③ 2-get-hubert-wav32k.py：生成 4-cnhubert/ 与 5-wav32k/"
# 注意变量名是 cnhubert_base_dir，不是配置文件里的 cnhuhbert_base_path（上游拼写不一致）
export cnhubert_base_dir="$GSV/GPT_SoVITS/pretrained_models/chinese-hubert-base"
run_py "2-get-hubert" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py

# -------------------------------------------------------------- ④ 语义 token
mark "④ 3-get-semantic.py：生成 6-name2semantic.tsv"
export pretrained_s2G="$GSV/$PRE_S2G"
export s2config_path="$S2_CONFIG"
run_py "3-get-semantic" GPT_SoVITS/prepare_datasets/3-get-semantic.py
printf 'item_name\tsemantic_audio\n' > "$OUT/6-name2semantic.tsv"
cat "$OUT/6-name2semantic-0.tsv" >> "$OUT/6-name2semantic.tsv"

for f in "$OUT/2-name2text.txt" "$OUT/6-name2semantic.tsv" "$OUT/4-cnhubert" "$OUT/5-wav32k"; do
  [ -e "$f" ] || { echo "!! 缺少 $f"; exit 1; }
done
echo "hubert 特征数: $(ls "$OUT/4-cnhubert" | wc -l)，32k wav 数: $(ls "$OUT/5-wav32k" | wc -l)"
du -sh "$OUT"

# 生成配置的两段 Python 只能从环境变量取参数：heredoc 里的 $ 会被 bash 先展开，
# 所以一律用引号版 <<'PYEOF'，把取值全部推到 os.environ。
export OUT EXP_NAME GPT_BS GPT_EPOCHS S2_BS S2_EPOCHS VERSION PRE_S1 PRE_S2G PRE_S2D

# s2_train 保存 checkpoint 时写的是 <exp_dir>/logs_s2_<version>/G_*.pth，
# 但脚本自己从不建这个目录 —— webui 在 open1Ba 里替它建了，我们没有。
# 训练全部跑完才在最后的 save_checkpoint 上炸，八轮白练。
mkdir -p "$OUT/logs_s2_$VERSION"

# ------------------------------------------------------------------ ⑤ 微调 GPT
mark "⑤ s1_train.py：微调 GPT（$GPT_EPOCHS epoch）"
S1_CFG="$OUT/tmp_s1.yaml"
"$VENV_PY" - "$S1_TPL" "$S1_CFG" <<'PYEOF'
import sys, yaml, os
src, dst = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(src, encoding="utf8"))
exp = os.environ["EXP_NAME"]; out = os.environ["OUT"]
d["train"]["precision"] = "32"                       # CPU 必须用 32
d["train"]["batch_size"] = int(os.environ["GPT_BS"])
d["train"]["epochs"] = int(os.environ["GPT_EPOCHS"])
# 只在最后一个 epoch 导出一次权重。上游默认每个 epoch 都导，
# v2 的 GPT 是 24 层，半精度导出约 150MB/个 —— 12 个 epoch 就是 1.8GB，
# 直接吃掉免费账号 500MB artifact 配额的三倍多。
d["train"]["save_every_n_epoch"] = int(os.environ["GPT_EPOCHS"])
# 只留最新一份带优化器状态的完整 ckpt（它比导出的权重大 5-6 倍，
# 不清的话 12 个 epoch 会在 runner 上堆出 ~10GB）
d["train"]["if_save_latest"] = 1
d["train"]["if_save_every_weights"] = True
d["train"]["if_dpo"] = False
d["train"]["exp_name"] = exp
d["train"]["half_weights_save_dir"] = os.environ["GPT_W"]   # 绝对路径，见文件头注释
d["data"]["num_workers"] = 2                          # runner 只有 2-4 核
d["pretrained_s1"] = os.environ["PRE_S1"]
d["train_semantic_path"] = out + "/6-name2semantic.tsv"
d["train_phoneme_path"] = out + "/2-name2text.txt"
d["output_dir"] = out + "/logs_s1"
yaml.safe_dump(d, open(dst, "w", encoding="utf8"), allow_unicode=True, sort_keys=False)
print("s1 config ->", dst)
PYEOF
run_py "s1_train" GPT_SoVITS/s1_train.py --config_file "$S1_CFG"

# --------------------------------------------------------------- ⑥ 微调 SoVITS
mark "⑥ s2_train.py：微调 SoVITS（$S2_EPOCHS epoch）"
S2_CFG="$OUT/tmp_s2.json"
"$VENV_PY" - "$S2_CONFIG" "$S2_CFG" <<'PYEOF'
import sys, json, os
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
exp = os.environ["EXP_NAME"]; out = os.environ["OUT"]
d["model"]["version"] = os.environ["VERSION"]
d["version"] = os.environ["VERSION"]
d["name"] = exp
d["data"]["exp_dir"] = out
d["s2_ckpt_dir"] = out
d["save_weight_dir"] = os.environ["SOV_W"]   # 绝对路径，见文件头注释
d["train"]["fp16_run"] = False                        # CPU：关闭 fp16 autocast
d["train"]["batch_size"] = max(1, int(os.environ["S2_BS"]) // 2)
d["train"]["epochs"] = int(os.environ["S2_EPOCHS"])
# 只在最后一个 epoch 导出一次。s2_train 的循环是 range(1, epochs+1)，
# 所以 save_every_epoch 设成 epochs 正好在最后一次命中（设成 0 会一次都不存）。
d["train"]["save_every_epoch"] = int(os.environ["S2_EPOCHS"])
d["train"]["if_save_latest"] = 1                      # 只留最新，省 runner 磁盘
d["train"]["if_save_every_weights"] = True
d["train"]["gpu_numbers"] = ""                        # s2_train 会读它，必须存在
d["train"]["pretrained_s2G"] = os.environ["PRE_S2G"]
d["train"]["pretrained_s2D"] = os.environ["PRE_S2D"]
d["train"]["grad_ckpt"] = False
d["train"]["lora_rank"] = 0
json.dump(d, open(dst, "w"), indent=2)
print("s2 config ->", dst)
PYEOF
run_py "s2_train" GPT_SoVITS/s2_train.py --config "$S2_CFG"

mark "微调产物校验"
# 这是整条训练链的闸门。s1/s2 都可能「训练跑完了但导出没发生」——
# 比如 s2 的 save_checkpoint 因为目录没人建而炸在最后一秒。
# 与其让 job 绿着把空目录传上去，不如在这里拦死并打印实际落盘的东西。
fail=0
for d in "$GPT_W" "$SOV_W"; do
  echo "--- $d ---"
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
    ls -lh "$d"
    du -sh "$d"
  else
    echo "  (空)"; fail=1
  fi
done
if [ "$fail" = "1" ]; then
  echo "!! 训练结束但没有导出任何权重。常见原因：" >&2
  echo "   - s2: <exp_dir>/logs_s2_<version>/ 目录不存在（save_checkpoint 会炸）" >&2
  echo "   - s1: half_weights_save_dir 没被创建" >&2
  echo "--- 训练中间目录 ---" >&2
  ls -la "$OUT" 2>/dev/null | head -20 >&2
  echo "--- logs_s2 ---" >&2
  ls -la "$OUT/logs_s2_$VERSION" 2>/dev/null | head -10 >&2
  exit 1
fi
echo "GPT 权重: $(ls "$GPT_W"/*.ckpt 2>/dev/null | wc -l) 个"
echo "SoVITS 权重: $(ls "$SOV_W"/*.pth 2>/dev/null | wc -l) 个"
