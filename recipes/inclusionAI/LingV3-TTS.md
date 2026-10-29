# LingV3-TTS

> Online serving for reference-conditioned voice synthesis and cloning

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/LingV3-TTS`
- Task: Text-to-speech with speaker-only and prompt-latent voice conditioning
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

## Notes

- Uploaded reference-audio voices are supported; built-in voice presets are not.
- Custom instructions and direct speaker-embedding input are not supported.
- Non-streaming output is mono 44.1 kHz WAV; streaming output is PCM.
