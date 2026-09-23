# R2D2 // Listening room

一个在 Apple Silicon Mac 上比较 Confucius4-R2T2 **官方 GGUF F16 / Q8_0 / Q4_K_M** 与 **自转 MLX BF16** 的本地流式识别网页。默认 GGUF Q8_0、160 ms 音频分块、中文识别；另可选英文、日文、韩文、西班牙文或自动识别。非中文语音可同时实时翻译为中文（腾讯 HY-MT1.5-1.8B，本机 CPU 运行）。

## 启动

本机环境、模型已经准备好。在项目目录运行：

```bash
./server.sh start     # 后台启动，就绪后打印本机与局域网地址
./server.sh status    # 运行状态、端口、模型引擎与加载状态
./server.sh log       # 日志最后 80 行；-f 实时跟随
./server.sh stop      # 停止，清理 llama-server 子进程，确认端口释放
```

打开 **http://localhost:8765**。服务默认监听 `0.0.0.0:8765`，`start` 会列出可用的局域网地址。

`stop` 先发 SIGTERM 让 FastAPI 的 lifespan 钩子正常关闭 GGUF 子进程；15 秒不退再强制结束，并扫掉指向本项目 `models/` 的 llama-server 残留（例如上次被 `kill -9` 留下的孤儿），最后确认端口已释放。

换端口：`PORT=8766 ./server.sh start`。只监听本机：`HOST=127.0.0.1 ./server.sh start`。换模型根目录：`R2D2_MODELS=/absolute/path ./server.sh start`。

局域网 HTTP 地址可以打开页面，但浏览器麦克风需要 `localhost` 或受信任的 HTTPS 页面。当前应用用于可信局域网内的单人测试，不包含账号系统。

## 使用

1. 默认 GGUF Q8，也可选 GGUF F16、GGUF Q4 或 MLX BF16；按「开始聆听」，允许浏览器使用麦克风。第一次加载与预热需要等待，状态栏会显示进度。
2. 白色文字已经确认，灰色尾字仍可能变化。停止后会处理积压音频并补齐尾字。
3. 切换模型，按「重放上一段」，以原录音速度把**完全相同的 PCM 音频**送给另一个模型。这里是重新识别，不会从扬声器播放声音。
4. 「用官方示例试跑」无需麦克风；「导入音频」支持当前浏览器能够解码的音频格式，最长 5 分钟 / 100 MB。
5. 在下方记录流畅度、错漏字、噪声误识别等主观听感；「导出记录」保存文字、参数、每步时序和听感笔记的 JSON。每条记录自带 `model`（权重组合）与 `model_detail`（从权重文件读出的量化类型与张量构成），不必回仓库对照；发生过解码器重建的，`resets` 列出时间点与原因，记录卡片上也会显示次数。音频留在当前页面内存中，不写入服务端磁盘，刷新页面即清空。
6. 「实时翻译」默认开启、识别语言为中文时自动隐藏。译文区与识别区同构：白字是已定稿的整句译文，只追加不回改；灰字是未完句的草稿译文，随识别更新而重译。识别确认一句后约 0.3–1 秒定稿。指标栏的 TRANSLATION LAG 是识别确认一句到该句译文定稿的时间。导出记录包含译文、每句原文 / 译文 / 耗时。策略与实测见 [docs/translation.md](docs/translation.md)。
7. 「热词与领域提示」即官方的 `context`：写进提示词的 system 段，用于人名、产品名等易错专有名词或一句话主题，上限 4000 字。它不是通用指令通道——实测写「翻译成英文」「只输出繁体」都会被模型忽略。

**声音处理**：默认开启浏览器降噪和回声消除，关闭自动增益，与调查时官方 demo 的麦克风参数一致。选择「原始输入」会请求关闭这三项。实际生效参数在导出的 `captureSettings` 中；浏览器或硬件仍可能有自身处理。导入文件与重放不重新应用麦克风处理设置。要比较降噪开关，需分别重新录音。

## 流式行为与比较边界

这不是把一段离线转写结果逐字播放。浏览器持续发送 16 kHz、单声道、PCM16 音频；后端持续执行识别步骤并回传文字。

两个引擎共用移植自官方 `streaming_transcribe_no_reset` 的调度：首步 320 ms（含 160 ms 前瞻），之后每步 160 ms；保留一个未确认 token；音频窗口超过 8 秒移除前 4 秒，并同步移除对应的提示文字；按中文 / 英文输出调整每步生成预算。每步重新编码当前音频窗口，并以已确认文字为续写前缀。标点按官方 `_normalize_punct_by_context` 归一：每个标点跟随前一个字符的字符集（汉字后全角、字母数字后半角）。

窗口为 8 秒 / 4 秒而非官方的 16 秒 / 8 秒：两个后端的单步成本都随窗口长度近似线性增长，这是这台机器上唯一的大杠杆。识别落后于输入时，一步最多合并 3 个 160 ms 分块一起解码，并按比例放大该步的 token 预算——音频编码与 prefill 每步只付一次，合并是真正能追上的手段。跟得上时仍是每 160 ms 一步。测量依据见 [docs/validation.md](docs/validation.md)。

官方发布的流式服务使用 NVIDIA / vLLM。本应用使用 llama.cpp / Metal 和 MLX，**不是官方 vLLM 服务的逐位复现**。不使用额外的人声检测模型、音量门限或服务端降噪。官方网页调查时 `use_vad=false`，但其开源服务仍有 FireRedVAD 计算。重复输出的处理上游分两层，本应用只取其一：检测到重复循环或单段超过 60 秒时按上游做法重建解码器状态（上游 `SOFT_RESET_SEC`），**已确认的文字一律保留、绝不回改**；上游那层会改写重复文本的 `detect_and_fix_repetitions` 不采用，因此模型真实的重复与幻觉仍然可见。停止时主动做一次尾部解码，修复整块边界上最后一个未确认 token 丢失的问题。这些差异及源码证据见 [docs/streaming.md](docs/streaming.md)。

延迟指标：

- **首字等待**：服务就绪后到首次确认文字的时间，包含输入音频本身的前置静音，不包含加载与预热。
- **单步识别耗时**：当前音频窗口的一次完整识别耗时，不等于说完某个字到显示的延迟。
- **待处理音频**：已收到但尚未完成解码的音频时长。连续超过 160 ms 的解码会积压；超过约 10 秒会明确报错停止，不丢帧掩盖延迟。

单次输入上限 5 分钟，只允许一个识别会话同时运行。切换时卸载上一模型，录音中禁止切换。

## 模型

**翻译：腾讯 HY-MT1.5-1.8B Q4_K_M**（1.13 GB，官方 GGUF，固定版本与 SHA-256 见 `models-manifest.json`）。在独立的 llama-server 中以 `-ngl 0 -t 6` 运行于 CPU：实测放在 GPU 上会让 Q8 识别 63% 的步骤超过 160 ms 预算，放在 CPU 上与单跑识别无差别。首次需要翻译的会话启动时加载（约 1 秒），之后常驻，内存约 1.2 GB。线程数可用 `R2D2_MT_THREADS` 调整。模型缺失时只关闭翻译，识别不受影响。

GGUF Q4 使用官方 **Q4_K_M 主权重 + Q8_0 音频投影**；官方没有发布 Q4 音频投影，因此与 Q8 选项共用同一份投影。导出记录用 `backend: gguf_q4` 标识，主权重和音频投影组合会一并记录。

GGUF Q8 使用官方 **Q8_0 主权重 + Q8_0 音频投影**，两者都量化；F16 选项继续使用原来的 F16 配对。Q8 文件从官方 Hugging Face 下载，固定版本和 SHA-256 记录在 `models-manifest.json`。导出记录用 `backend: gguf_q8` 标识 Q8，并附带模型说明；旧的 `backend: gguf` 始终表示 F16。

GGUF F16 与 MLX BF16 从另一台本机复制到本项目 `models/`，Q8_0 配对及 Q4_K_M 主权重从官方 Hugging Face 下载；**运行时不联网，也不下载 HF 原始模型**。主要权重的校验记录见 [models-manifest.json](models-manifest.json)。模型、虚拟环境、`artifacts/` 与 `results/` 中的本机测试输出均不纳入 Git。

**选型结论：GGUF Q8_0。** 两轮远场实测中 Q8 与 F16 无法区分、单步耗时剖面最好，Q4 明显更差（依据见 [docs/validation.md](docs/validation.md)）。llama-server 以 `-c 4096 --cache-ram 0` 启动：每步音频都不同，默认 8 GiB 的 prompt 缓存无法复用只会占满内存；关闭后 Q8 连续运行的进程内存约 2.7 GB。

```text
models/
├── Confucius4-R2T2-GGUF/
│   ├── Confucius4-R2T2-f16.gguf
│   ├── mmproj-Confucius4-R2T2-f16.gguf
│   ├── Confucius4-R2T2-Q4_K_M.gguf
│   ├── Confucius4-R2T2-Q8_0.gguf
│   └── mmproj-Confucius4-R2T2-Q8_0.gguf
└── Confucius4-R2T2-MLX-BF16/
    ├── model.safetensors
    └── 配置、特征提取器、分词器文件
└── HY-MT1.5-1.8B-GGUF/
    ├── HY-MT1.5-1.8B-Q4_K_M.gguf
    └── License.txt
```

GGUF 适配器复用 MLX 目录中的同源分词器执行尾字回退，因此两份目录都要保留。默认 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`。GGUF 内部推理服务仅监听随机的本机回环端口，网页由同一个 FastAPI 服务提供。

## 在另一台 Apple Silicon Mac 上安装

需要 Python 3.12、[uv](https://docs.astral.sh/uv/) 和 llama.cpp。当前验证的 llama.cpp 是 `0.4.1 / b29c606e2`，mlx-audio 是 `0.5.5`。

```bash
brew install llama.cpp
uv sync --frozen
# 将两份模型目录放到 models/ 后：
./server.sh start
```

## 开发与验证

无需前端构建，原生 JavaScript + AudioWorklet；Python FastAPI + WebSocket，模型计算在专用单线程执行器里运行，避免阻塞收音。

```bash
uv run pytest -q
node tests/test_worklet.cjs
# 先启动服务；以下脚本按真实音频时钟发送官方示例：
uv run python scripts/smoke.py --backend gguf --output artifacts/gguf-smoke.json
uv run python scripts/smoke.py --backend mlx --output artifacts/mlx-smoke.json
uv run python scripts/smoke.py --backend gguf_q8 --output artifacts/gguf-q8-smoke.json
uv run python scripts/smoke.py --backend gguf_q4 --output artifacts/gguf-q4-smoke.json
# 识别 + 实时翻译（任一非中文 16 kHz 单声道 WAV）：
uv run python scripts/smoke.py --backend gguf_q8 --audio artifacts/mt/en.wav --language English --translate --output artifacts/mt/en-q8.json
# 20.22 秒，覆盖 8 秒滚动窗口边界：
uv run python scripts/smoke.py --backend mlx --repeat 3 --output artifacts/mlx-long.json
```

测试覆盖分块边界、尾音刷新、中文 token 回退、自动语言、英文空格、窗口移动、积压时的分块合并（不丢音频、按块数放大预算、跟得上时不合并）、模型独占、非法输入，标点上下文归一（按前字选全半角、长度不变且幂等、以前缀为上下文）、重复循环检测与解码器重建（已确认文字不改、重建后不在同一段尾巴上反复触发）、60 秒安全重建、停止时不重建、六个语言选项与非法语言拒绝、热词 4000 字上限，16 / 44.1 / 48 kHz 麦克风重采样，以及翻译的切句（句点须后接空白、长句按分句标点切）、中文直通、定稿只追加且保持句序、慢翻译跳过过期草稿、草稿复用、定稿时清除覆盖该句的草稿、翻译故障不影响识别、中文会话不加载翻译模型。实测结果与限制见 [docs/validation.md](docs/validation.md)。

界面遵循 Bearbone Design System v0.2 Dark：纯黑底、黑白灰、等宽字体、细线双栏、事实页脚。Berkeley Mono / Sarasa Mono SC 使用本地字体及回退，不包含商业字体文件。

## 上游来源

- [官方模型](https://huggingface.co/netease-youdao/Confucius4-R2T2)
- [官方 GGUF](https://huggingface.co/netease-youdao/Confucius4-R2T2-GGUF)
- [官方 demo](https://r2t2.youdao.com/demo)
- [官方源码](https://github.com/netease-youdao/Confucius4-R2T2)
- [HY-MT1.5-1.8B GGUF](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF)（翻译）

上游代码与示例音频的来源、许可见 [third_party/NOTICE.md](third_party/NOTICE.md)。模型权重遵循各自随附的网易模型许可，本项目不重新授权模型。
