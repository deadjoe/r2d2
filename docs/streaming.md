# 官方 streaming 调查与本地适配

调查日期：2026-09-22。阅读了实际 Python 服务、模型流式方法和线上 demo 的浏览器脚本，而不仅是 README。

参考源码快照：`netease-youdao/Confucius4-R2T2@80c22e6140bcb9166fb9906798894fc8b18c8309`。原件来自用户 benchmark 的干净 upstream checkout。

## 已核实的实现

1. [`ws_server.py`](https://github.com/netease-youdao/Confucius4-R2T2/blob/80c22e6140bcb9166fb9906798894fc8b18c8309/ws_server.py) 的 `/asr_stream_api_v1` 接收 PCM16，固定 160 ms 分块；第一步等待 160 ms 额外音频。模型运行在 vLLM。
2. [`r2t2/r2t2_asr.py`](https://github.com/netease-youdao/Confucius4-R2T2/blob/80c22e6140bcb9166fb9906798894fc8b18c8309/r2t2/r2t2_asr.py) 中 `streaming_transcribe_no_reset` 将当前窗口的全部音频重新输入模型，以已确认文字为 assistant 的续写前缀。它不是音频编码器跨包复用内部缓存的实现。每次只生成少量 token，默认保留末尾 1 个 token，避免过早确认。
3. 累积音频超过 16 秒移除前 8 秒，官方以首轮 49 块、后续 50 块来同步裁剪文字列表。本应用以各块实际音频结束位置做同等裁剪，并按下文所述把窗口改为 8 秒 / 4 秒。
4. 服务端每步预算从 4 开始；有新字后基准为 2；中文乘 2；英文无新字时每步加 0.5；默认上限 4。
5. 自动语言通过模型输出 `language X<asr_text>` 解析，指定语言则直接在 assistant 前缀中提供语言标签。
6. [线上 demo](https://r2t2.youdao.com/demo) 的麦克风请求为 `noiseSuppression: true`、`echoCancellation: true`、`autoGainControl: false`，发送 160 ms 音频包。调查时页面配置 `mode: slow`、`use_vad: false`、`smooth: false`。线上后台的实际部署版本无法仅凭网页源代码证明与开源仓库完全一致。
7. 开源 v1 服务仍调用 FireRedVAD，但只有 `use_vad` 开启时才依检测结果切断上下文；并有重复字符 / 重复短语检测与模型状态重置。这是另一层处理，不能把其作用都归于 ASR 模型。

## 两个后端

| 项目 | GGUF | MLX |
| --- | --- | --- |
| 模型 | 官方 F16 / Q8_0（默认）/ Q4_K_M | 用户自转 BF16 |
| 计算 | llama.cpp 0.4.1，Mac 上 Metal，Linux 上 CUDA | mlx-audio 0.5.5，MLX / Metal，仅 Mac |
| 输入 | 原始聊天前缀 + WAV，`/completion` 多模态接口 | Whisper 特征提取 + 音频编码 + 拼接 assistant 前缀 |
| 输出 | 贪心生成 2–4 token 后统一回退 | 同样贪心生成后统一回退 |
| 调度 | 同一个 `r2d2/streaming.py` | 同一个 `r2d2/streaming.py` |

GGUF 使用 llama.cpp `/props` 返回的 media marker，不能把随机化的 marker 写死；通过 `/completion` 的 `prompt_string` / `multimodal_data` 传入音频。参考[对应版本的服务文档](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/tools/server/README.md)。

MLX 不使用 `generate(stream=True)` 来冒充连续收音。该接口只负责给定音频的 token 输出。本应用直接构造音频与文字嵌入，把已确认文字加入 assistant 前缀，再调用生成器；`mlx-audio` 的 `Model` 包装类需解包到 `_model`，否则嵌入接口的签名检查失败。

## 有意保留的差异

- 后端、精度、音频特征计算及提示缓存不同，不能声称与官方 vLLM 数值一致或达到官方延迟。
- 默认中文，官方网页调查时为英文；可以在界面切换。
- 展示灰色的未确认尾字；官方网页调查时 `show_partial=false`，主要展示确认文字。
- 停止时额外做一次最多 16 token 的解码，不补静音；即使停止位置正好在块边界，也释放未确认尾字。官方 finish 在没有残余 buffer 时直接返回，存在尾字不刷新的边界。
- 不加入 FireRedVAD 或人声门限。保留静音和噪声下真实识别输出，便于人工判断；浏览器降噪开关与该层分开。
- 重复处理只取上游两层中的一层：`detect_hallucination` 的尾部循环检测（与上游一样只查上次重建以来确认的文字，上游对应 `last_fixed_asr_text` 在重建时清空；若查全文，同一段尾巴会在重建后每步重复命中），命中后按上游做法重建解码器状态（丢弃窗口音频与前缀，已确认文字保留不改）；无 VAD 时单段超过 60 秒也重建，对应上游 `SOFT_RESET_SEC`。会改写文本的 `detect_and_fix_repetitions` 不采用，所以模型的重复与幻觉在输出里仍然可见，只是不再自我喂养。重建会在该条更新的 `reset` 字段报出。
- 预算加倍的条件从「中文」扩大为「字符密集的文字」：汉字、假名、谚文。上游只判断汉字，日文借汉字碰巧通过，韩文则落到英文预算（每步净确认约 1 个 token），快速对话时文字落后超过 4 秒，窗口裁掉的语音就永久丢失。在 `scripts/testset.py` 的多人对话测试集上（Q8，参考为人工韩语字幕）：离线同段 A/B 韩语 ko-2 错误率 39.3% → 22.7%，漏字 472 → 140；实时复测 ko-1 29.2% → 23.1%、ko-2 39.3% → 23.1%，文字延迟 p50 从 1.1–1.9 秒降到 0.3–0.4 秒。日文、英文不受影响。代价是韩语每步多生成约 0.6 个 token（约 3 ms）。
- 标点按上游 `_normalize_punct_by_context` 归一：每个标点跟随其前一个非空白字符的字符集（汉字后全角、ASCII 字母数字后半角）。该变换长度不变且幂等，因此不影响按位置切分的已确认文本。
- 不做官方的中英文标点转换后处理，输出模型本身产生的标点；中英文空格仅作相邻汉字间空格的归一化。
- 滚动窗口为 8 秒 / 4 秒，不是官方的 16 秒 / 8 秒。此前本机测量显示单步成本随窗口近似线性增长（GGUF 约 0.77 ms / 音频 token，MLX 约 0.66 ms / token，另加随窗口增长的音频编码）；16 秒窗口下逐个处理 160 ms 分块无法持续跟上输入。用户已采纳方案 A，优先保留常规 160 ms 推进粒度。现存 87.62 秒实时记录中，MLX 少识别 2 个字（258 / 260）；前次报告的 16 秒离线对照为 260 / 260，但未留存可供本次复核的原始输出，不能据此断言一般场景的精度损失。证据边界与取舍见 [validation.md](validation.md)。
- 落后于输入时一步最多合并 3 个分块（480 ms）一起解码，并把该步 token 预算按合并块数放大。音频编码与 prefill 每步只付一次，所以合并能真正追上；跟得上时仍是每 160 ms 一步，与官方一致。这是有意加入的降级路径，代价是落后时文字更新间隔变长。
- 仍然不丢包、不跳过音频。合并只改变一步覆盖多少音频，不丢弃任何采样；积压仍然如实上报。

## 手动比较建议

分别录制安静讲话、远距离讲话、键盘 / 风扇声、无讲话环境声、句间停顿和中英混说。每段先选一个后端录音，再在另一个后端重放同一段。需要比较浏览器降噪时重新录音，原始录音和处理后录音应分开记录。页面笔记用来记录你听到 / 看到的差异，不自动给两种模型打分。

## llama-server 参数

- `--cache-ram 0`：每步都带新音频，保存的 prompt 状态永远无法复用；默认 8 GiB 主机内存 prompt 缓存只会被填满。上游 `r2t2_llama` 原生后端每次调用前 `llama_memory_clear` 同理。实测 Q8 连续 270 秒：RSS 11.9 GB → 2.7 GB，400 步逐步输出完全一致，单步耗时不变。
- 提示词与模型自带 `chat_template.json` 一致：system 文本后直接接 `<|im_end|>`，不加换行（mlx-audio 多加的换行是它自己的偏差）；与上游 `build_asr_prompt` 逐字节一致。
