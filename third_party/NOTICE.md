# Third-party notices

`r2d2/streaming.py` adapts the streaming scheduling ideas and behavior of
`r2t2/r2t2_asr.py` and `ws_server.py` from:

- Project: https://github.com/netease-youdao/Confucius4-R2T2
- Revision: `80c22e6140bcb9166fb9906798894fc8b18c8309`
- Copyright 2026 The NetEase Youdao team.
- License: Apache License, Version 2.0; see `Confucius4-R2T2-LICENSE`.
- Adaptations: backend-independent runtime, audio-position-based window trimming,
  explicit final flushing, local GGUF and MLX integrations. Also ported:
  `_normalize_punct_by_context`, `detect_hallucination` and the decoder rebuild on
  a loop or after `SOFT_RESET_SEC`. Not included: the vLLM backend, FireRedVAD,
  and the text-rewriting `detect_and_fix_repetitions`.
- Both source files are unchanged at the later revision
  `c4611929bc3592b38dab34e96a8c9940d6da3755` (verified 2026-09-23), which adds
  `r2t2_llama/`; its prompt construction and per-call memory clearing informed
  `r2d2/backends.py`, but no code from it is copied.

`tests/fixtures/official-test.wav` is the repository's `resources/test.wav`
at the same revision, included for reproducible local integration tests.

The model files are excluded from version control. They retain their own
NetEase Model Use License Agreement; see the model's included `MODEL_LICENSE`
and https://huggingface.co/netease-youdao/Confucius4-R2T2-GGUF .

No commercial Berkeley Mono font files are distributed.

The live translation model, `tencent/Hy-MT2-1.8B-GGUF` (Q4_K_M), is also
excluded from version control and is used unmodified through llama.cpp. It is
licensed under the Apache License 2.0, downloaded next to the weights as
`models/Hy-MT2-1.8B-GGUF/LICENSE.txt`. `r2d2/translate.py` uses the prompt
templates from the model card; no Tencent code is copied.
