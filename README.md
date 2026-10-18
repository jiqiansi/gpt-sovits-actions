# gpt-sovits-actions

用 **GitHub Actions 的免费 CPU runner** 跑通 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)：
装依赖 → 下预训练权重 → 用一小段音频微调出一个新音色 → 合成语音 → 把 wav 作为 artifact 下载回来。

**全程没有显卡。** 从点 Run workflow 到拿到 wav，一个 job 走完。

---

## 一、能跑什么

| mode | 做什么 | 大约耗时 | 产物 |
|---|---|---|---|
| `infer` | 直接用预训练权重合成 | ~15 分钟 | `tts-output-*` artifact（wav） |
| `train` | 只做微调，产出 GPT + SoVITS 两套权重 | ~60–120 分钟 | `finetuned-weights-*` artifact |
| `full` | 微调后再用新权重合成 | ~90–150 分钟 | 两个 artifact 都有 |

耗时是变量，主要看 `gpt_epochs` / `s2_epochs` 和训练集长度。

---

## 二、怎么用

1. 打开仓库 → **Actions** → 选 **GPT-SoVITS · CPU 跑通** → **Run workflow**。
2. 表单里填：
   - `mode`：第一次建议先跑 `infer`，确认整条链路是通的；再跑 `full`。
   - `text`：要合成的目标文本（中文）。
   - `gpt_epochs` / `s2_epochs`：数据少时别贪多，`12` / `8` 够了。
   - `hf_mirror`：网络到 huggingface.co 不稳就勾上，改走 hf-mirror.com。
3. 跑完在 run 页面底部 **Artifacts** 下载。
   - `tts-output-<mode>-<version>`：里面有 `tts_out.wav` 和同名 `.json`（记录用了哪两个权重、采样率、时长）。
   - `finetuned-weights-<version>`：`*-e*.ckpt`（GPT）和 `*_e*_s*.pth`（SoVITS）。

**别只看 job 是绿的。** `校验成片` 这一步会用 ffprobe 检查时长和体积，但下载回来还是建议自己听一遍。
本仓库的判断标准：成片时长 > 1s、体积 > 200KB，并且和本地手跑的成片同量级。

### 自己换训练音频

把 `data/raw/train/` 下的 wav 换成你自己的录音，并同步改 `data/train_manifest.json` 的 `text`。

> **`text` 必须和音频里说出来的字一字不差。** 对不上的静音片段会被静默跳过，
> 表现为「训练好像成功了但音色不对」。这是 GPT-SoVITS 最常见的翻车点。

参考音频 `data/raw/ref.wav` 必须落在 **3–10 秒**之间 —— 这是上游硬性限制，
超了会直接 `OSError: 参考音频在3~10秒范围外`。对应的提示文本放在 `data/ref_text.txt`。

本仓库自带的音频是用 gTTS 合成的演示数据（约 17 秒，两句话）。切点不是估的：
用 `ffmpeg -af silencedetect` 找到了实测停顿位置，在「参考音频。」后面的静音处下刀，
所以 `01.wav` / `02.wav` 的文本和音频能严格对齐。

---

## 三、为什么要自己写这些脚本

上游 `install.sh` 是给本地 conda 环境用的，直接搬到 CI 上有几个硬伤：

| 上游做法 | 在 CI 上的问题 | 本仓库的处理 |
|---|---|---|
| 要求先装 conda | runner 上只有 base 环境（3.12），GPT-SoVITS 要 3.10 | `python3.10 -m venv`，省掉 conda 的 solver 和几 GB 磁盘 |
| 下 4.56GB `pretrained_models.zip` | 里面 2/3 是 v3/v4/v5/v2Pro/BigVGAN/uVR5/faster-whisper，全用不上 | `fetch_models.sh` 只下 7 个文件，约 **1.6GB** |
| 装 PyPI 默认的 torch | 那是 CUDA 版（2GB+），无 GPU runner 上纯浪费 | 显式 `--index-url https://download.pytorch.org/whl/cpu`（**174MB**） |
| `requirements.txt` 里 x86_64 装 `onnxruntime-gpu` | 多下几百 MB，且 runner 上没有 GPU | 换成 CPU 版 `onnxruntime` |
| 同文件里 `--no-binary=opencc` | 在 CI 上会硬编译 | 吃现成 wheel |
| 无重试 | CI 单次 TLS 断流很常见，一次抖动就红 | 所有下载都套 5 次重试 + 长超时 |

### 三处会让人查很久的坑

1. **`cnhubert_base_dir` ≠ `cnhuhbert_base_path`**
   `2-get-hubert-wav32k.py` 读的环境变量叫 `cnhubert_base_dir`，而配置文件里拼写是
   `cnhuhbert_base_path`（h u h）。按配置文件的名字传会静默取不到模型。

2. **`s2_train.py` 必须给 `gpu_numbers`**
   脚本第一行就是 `os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")`，
   CPU 上没人会想到要填这个字段，缺了直接 `AttributeError`。空字符串就行。

3. **CPU 上要显式关掉半精度**
   - `is_half=False`（prepare 脚本会再和 `torch.cuda.is_available()` 取与，但显式写死更安全）
   - s1 的 `precision` 改成 `"32"`：写 `16-mixed` 时 pytorch-lightning 会**静默**换成
     `bf16-mixed`，既慢又可能丢精度
   - s2 的 `fp16_run=False`

---

## 四、仓库结构

```
.github/workflows/gpt-sovits.yml   # 唯一的 workflow，workflow_dispatch 触发
scripts/
  bootstrap.sh      # 装 python3.10 venv + CPU torch + 依赖（含 5 次重试）
  fetch_models.sh   # 按需下预训练权重，1.6GB（不是上游的 4.56GB）
  make_list.py      # data/train_manifest.json -> logs/<exp>/dataset.list
  train.sh          # ①音素+BERT ②HuBERT ③语义token ④s1 ⑤s2，全 CPU
  tts_synthesize.py # 参考音频 + 目标文本 -> wav
data/
  raw/ref.wav       # 参考音频（5.5s，3~10s 硬约束）
  raw/train/*.wav   # 训练集
  ref_text.txt      # 参考音频对应的文字
  train_manifest.json  # 训练集清单：wav + 逐字对齐的文本
```

运行期产物全部在 `.gitignore` 里：`.venv/`、`GPT-SoVITS/`、`logs/`、`artifacts/`、`GPT_weights*/`、`SoVITS_weights*/`。

---

## 五、几条约束（省得你白等）

- **免费 runner 没有 GPU**，所以本仓库整条链路都是 CPU 版。想要更快得自己接
  self-hosted GPU runner，`train.sh` 里的 `is_half=False` 和 `precision=32` 要相应改回半精度。
- **单个 job 上限 6 小时**。当前配置（12+8 epoch、两条训练音频）远没到这个数，
  但如果你把训练集换成几十分钟的音频 + 几百 epoch，就会撞线。
- **artifact 默认 14 天过期**，要长期留记得下载走。
- **公开仓库的 Actions 分钟数是免费的**，私有仓库每月额度有限。
- `version` 目前只实现了 `v1` / `v2`。v3/v4/v5/v2Pro 需要额外的 vocoder / sv 权重和
  不同的训练脚本（`s2_train_v3_lora.py`），没在这条流水线里。
