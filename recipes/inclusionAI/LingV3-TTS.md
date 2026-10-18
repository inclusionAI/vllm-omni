# LingV3-TTS

> Online serving for voice cloning, instruction-conditioned speech and Podcast

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/LingV3-TTS`
- Task: Text-to-speech with reference voices, instructions and multiple speakers
- Mode: Online serving via the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: Community

## References

TO-FILL-IN

## Hardware Support

LingV3-TTS currently uses a two-stage CUDA/BF16 pipeline.
The default single-GPU configuration targets one H100-class GPU with 80 GB of memory.

## CUDA

### 1x H100 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- CUDA Driver Version: 590.48.01
- CUDA: 13.0
- vLLM version: 0.28.0
- vLLM-Omni version or commit: 262ff14

#### Command

From the vLLM-Omni repository root, launch with

```bash
vllm serve /path/to/lingv3_tts \
  --omni \
  --port 8000
```

#### Verification

The vendor example `seed_zh_prompt.wav` is byte-identical to the public
Seed-TTS Chinese prompt `10002287-00000094.wav`. Download the copy published
by inclusionAI if a local reference is preferred:

```bash
curl -L \
    https://raw.githubusercontent.com/inclusionAI/Ming-omni-tts/main/data/wavs/10002287-00000094.wav \
    --output ./seed_zh_prompt.wav
```

##### Speaker-only conditioning

Provide one reference clip and omit `ref_text`:

```bash
curl http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "/root/models/inclusionAI/lingv3_tts_0805",
      "input": "今天天气很好，我们一起出去散步吧。",
      "ref_audio": "https://raw.githubusercontent.com/inclusionAI/Ming-omni-tts/main/data/wavs/10002287-00000094.wav",
      "response_format": "wav",
      "max_new_tokens": 200
    }' --output lingv3_speaker_only.wav
```

##### Prompt-latent conditioning

Add the exact reference transcript:

```bash
curl http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "/root/models/inclusionAI/lingv3_tts_0805",
      "input": "简单地说，这相当于惠普把消费领域市场拱手相让了。",
      "ref_audio": "https://raw.githubusercontent.com/inclusionAI/Ming-omni-tts/main/data/wavs/10002287-00000094.wav",
      "ref_text": "在此奉劝大家别乱打美白针。",
      "response_format": "wav",
      "max_new_tokens": 200,
      "seed": 1234
    }' --output lingv3_prompt_latent.wav
```

##### Reusing an uploaded voice

Upload a reference clip once, then use its name instead of sending `ref_audio` on every request. Include `ref_text` to use prompt-latent conditioning; omit it for speaker-only conditioning.

```bash
curl -X POST http://localhost:8000/v1/audio/voices \
  -F "audio_sample=@./seed_zh_prompt.wav" \
  -F "consent=user-consent-id" \
  -F "name=lingv3_demo" \
  -F "ref_text=在此奉劝大家别乱打美白针。"

curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/models/inclusionAI/lingv3_tts_0805",
    "input": "简单地说，这相当于惠普把消费领域市场拱手相让了。",
    "voice": "lingv3_demo",
    "response_format": "wav",
    "max_new_tokens": 200
  }' --output lingv3_uploaded_voice.wav
```

## Additional conditioning and sampling

Use a checkpoint that implements the requested task. SFT bundles declaring
`tts_config.task_contract=lingv3_sft_multitask_v1` retain their KDA gate
configuration. Checkpoint `sampling_temperature`, `minimum_decode_steps` and
`maximum_decode_steps` provide the Flow sampling and decode defaults.

### Instructions and speech without a reference

`instructions` accepts freeform text, a JSON string containing short controls,
or a complete single-speaker `audio_sequence` JSON string. Short controls are
`语速`, `基频`, `音量`, `情感`, `方言`, `风格` and `IP`. A complete caption must
contain the single `speaker_1` item and inactive BGM fields; its field order is
preserved when tokenizing. The assistant-side instruction terminator follows
the checkpoint tokenizer EOS.

For controlled speech without a reference, omit `ref_audio`, `ref_text` and
`voice`, and provide an instruction:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/lingv3_tts",
    "input": "清晨的风穿过树林，带来了远方的消息。",
    "instructions": "{\"情感\":\"高兴\"}",
    "response_format": "wav",
    "seed": 1234
  }' --output lingv3_controlled.wav
```

This mode retains a masked speaker slot. Omitting the instruction as well
selects emergent speech with no speaker slot. Reference-conditioned requests
may also include instructions. Set `language` to a target language supported
by the checkpoint; aliases such as `zh`, `en` and `ja` are normalized before
prompt construction.

### Podcast

Podcast requires a checkpoint declaring
`tts_config.task_decode_alignment.podcast=old_shift`. Provide `ref_audio` as
an ordered list, one clip per speaker. `ref_text` must contain exactly one
ordered `speaker_N: text` line per clip. Each nonempty input line represents
one turn, and the input must use every supplied speaker. Podcast does not
accept `instructions`.

For a server launched with `--allowed-local-media-path /path/to/references`:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/lingv3_tts",
    "input": "speaker_1:你好，今天我们聊聊语音合成。\nspeaker_2:好呀，我们开始吧。",
    "ref_audio": [
      "file:///path/to/references/speaker_1.wav",
      "file:///path/to/references/speaker_2.wav"
    ],
    "ref_text": "speaker_1:第一段参考音频的准确文本。\nspeaker_2:第二段参考音频的准确文本。",
    "language": "Chinese",
    "response_format": "wav",
    "seed": 1234
  }' --output lingv3_podcast.wav
```

The adapter derives Podcast decode bounds from the turn text; an explicit
`max_new_tokens` can tighten the upper bound but must not fall below the
derived minimum. When target-language conditioning is enabled, supply
`language` unless the checkpoint also enables language inference from text.

### Sampling controls

Use `extra_params` for explicit Flow sampling overrides:

```json
{
  "extra_params": {
    "temperature": 1.5,
    "lingv3_cfg": 1.0
  },
  "seed": 1234
}
```

Both controls must be finite non-negative numbers. Omitting `temperature`
preserves the checkpoint Flow temperature; explicitly setting `0.0` overrides
it. `lingv3_cfg` defaults to `1.0` and uses the shared formula
`cond + cfg * (cond - uncond)`. Compare CFG formulas before comparing numeric
settings across implementations. Supply `seed` explicitly for reproducible
comparisons with a fixed checkpoint and environment.

## Notes

- Uploaded reference-audio voices are supported; built-in voice presets are not.
- Uploaded voice versions and reference content identify reusable speaker
  embeddings; prompt latents remain request-specific.
- Reference clips retain the serving endpoint's 1–30 second duration limits.
- Direct speaker-embedding input is not supported by the speech adapter.
- Non-streaming output is mono 44.1 kHz WAV; streaming output is PCM.
