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

### 八个会让人查很久的坑

都是真踩过的，每条对应一次 CI 翻车。

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

4. **torch 必须 ≥ 2.6**
   transformers 的 `check_torch_load_is_safe()` 直接卡这一条（CVE-2025-32434），
   低于它连 `chinese-roberta` 的 `pytorch_model.bin` 都不肯读，推理第一步就死。
   上游 README 那张表里的 2.5.1 只是「测试过的组合」，不是硬约束。

5. **`resampy` 不在上游 requirements.txt 里**
   `_set_prompt_semantic` 硬编码 `match_librosa=True`，参考音频 32k→16k 那步必然
   `import resampy`。Windows 整合包里带了，所以本地发现不了。

6. **nltk_data 必须落在 `sys.prefix/nltk_data`**
   nltk 只查 `NLTK_DATA` / `~/nltk_data` / `sys.prefix/nltk_data` 等少数几处，
   **没有 cwd 相对路径**。解到 GPT-SoVITS 目录里等于白解 —— 文件都在，就是没人去找。
   目标文本里只要出现拉丁字母（`GitHub`、`CPU`），LangSegmenter 就会切出英文段，
   于是 `LookupError: averaged_perceptron_tagger_eng`。

7. **配置模板在 `GPT_SoVITS/configs/`，不是 `configs/`**
   仓库根没有 `configs/` 这一层。写错的话要等前三个 prepare 脚本跑完才报
   `FileNotFoundError`，白等几分钟还不好定位。

8. **`s2_train.py` 不会自己建 `logs_s2_<version>/`**
   它把 checkpoint 存到 `<exp_dir>/logs_s2_<version>/G_*.pth`，但目录得外面建
   （webui 在 `open1Ba` 里替它建了）。s1 跑完 12 epoch、s2 跑完 8 epoch，
   全在最后一秒的 `shutil.move` 上炸掉，导出权重一步没走到。

> 另外两条「闸门失灵」的教训，跟 GPT-SoVITS 无关但同样致命：
> `grep file > out` 在文件不存在时会安静地产出空文件，pip 对空清单照样返回成功；
> `if cmd | tail; then` 拿到的是 `tail` 的退出码，永远为真。
> 本仓库所有校验都显式检查退出码，不依赖 `set -e`。

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
  push_via_api.py   # git push 走投无路时，用 Git Data API 推
  watch_run.py      # 轮询一次 run 的每一步进度
data/
  raw/ref.wav       # 参考音频（5.5s，3~10s 硬约束）
  raw/train/*.wav   # 训练集
  ref_text.txt      # 参考音频对应的文字
  train_manifest.json  # 训练集清单：wav + 逐字对齐的文本
```

运行期产物全部在 `.gitignore` 里：`.venv/`、`GPT-SoVITS/`、`logs/`、`artifacts/`、`GPT_weights*/`、`SoVITS_weights*/`。

---

## 五、实测数据与硬约束

**runner 实测**（公开仓库，2026-10-18，`ubuntu-24.04`）：

| 项 | 实测值 |
|---|---|
| CPU / 内存 | 4 vCPU（AMD EPYC）/ 16 GiB |
| 磁盘 | 145 GB，作业开始时约 86 GB 可用 |
| 装依赖 | 131–178 s（含 187MB CPU 版 torch、现场编译 pyopenjtalk/jieba_fast） |
| 下权重 | 22–33 s（1.1 GB 预训练 + 635 MB G2PW） |
| s1 微调 12 epoch | 约 13 min（77.6M 参数，50 step/epoch） |
| s2 微调 8 epoch | 约 21 min（100 step/epoch，1.6 s/it） |
| 合成 6 秒音频 | 约 1–2 min |

> 私有仓库的 runner 是 2 vCPU / 8 GB / 约 14 GB 可用，磁盘差 5 倍。
> 想跑得快就保持公开。**ML 训练在 GitHub 的使用条款里是允许的**（禁止的是加密货币挖矿）。

- **单个 job 上限 6 小时**（不可调）。当前 12+8 epoch 约 42 分钟，余量很大。
  换成几十分钟的训练集 + 几百 epoch 就会撞线。
- **免费账号 artifact 存储只有 500 MB，且与 GitHub Packages 共用**。
  这就是「只在最后一个 epoch 导出一次权重」的原因 —— v2 的 GPT 半精度约 150 MB/个，
  每个 epoch 都导，12 个就是 1.8 GB，直接超配额 3 倍。
- **artifact 保留期设成 7 天**，缩短配额被占用的周期。要长期留记得下载走。
- **`ubuntu-latest` 从 2026-10-19 起往 26.04 迁**（默认 Python 3.12 → 3.14），
  所以 workflow 钉死 `ubuntu-24.04`。镜像一换，最可能先炸的就是装依赖那步。
- `version` 目前只实现了 `v1` / `v2`。v3/v4/v5/v2Pro 需要额外的 vocoder / sv 权重和
  不同的训练脚本（`s2_train_v3_lora.py`），没在这条流水线里。

### 推送：git push 走投无路时的兜底

本机直连 `github.com` 会间歇性断、代理也常挂，而 `api.github.com` 往往还通着。
`scripts/push_via_api.py` 就是为这个场景写的：把一次 commit 拆成
blob → tree → commit → ref 四个 API 对象推上去，文件按字节 base64 传。

```bash
git commit -am "..."
python3 scripts/push_via_api.py "commit message"
```

注意它不能删文件（远端多出来的文件会提示，不会动）。真要删，等网络正常时用 git push。

