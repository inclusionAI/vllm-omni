# LingV3 TTS online inference

Run from the vLLM-Omni repository root with a compatible LingV3 checkpoint.

## Start the server

```bash
MODEL=/path/to/LingV3-Tiny-8B-MoE-HF \
  bash examples/online_serving/text_to_speech/lingv3_tts/run_server.sh
```

## Generate speech

The online client and offline inference example default to seed `1234`.
Use `--seed` to override it.

```bash
python examples/online_serving/text_to_speech/lingv3_tts/speech_client.py \
  --text '雨后的山谷很安静，远处亮起了一盏温暖的灯。' \
  --ref-audio /path/to/prompt.wav \
  --ref-text '参考音频对应的准确文本。' \
  --output outputs/seed.wav
```

For speech without a reference voice:

```bash
python examples/online_serving/text_to_speech/lingv3_tts/speech_client.py \
  --text '清晨的风穿过树林，带来了远方的消息。' \
  --output outputs/emergent.wav
```

Use `--instructions '{"情感":"高兴"}'` for controlled speech. For Podcast,
repeat `--ref-audio` in speaker order and use matching `speaker_N:` lines in
`--text` and `--ref-text`.
