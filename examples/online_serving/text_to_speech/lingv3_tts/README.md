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

## Stream raw PCM

Start the server with `--allowed-local-media-path /path/to/reference_audio` to
allow local reference audio, then request raw audio streaming:

```bash
BASE_URL=http://127.0.0.1:8091
mkdir -p outputs/online
curl -fsS --no-buffer "$BASE_URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "lingv3-tts",
    "input": "雨后的山谷很安静，远处亮起了一盏温暖的灯。",
    "ref_audio": "file:///path/to/reference_audio/prompt.wav",
    "ref_text": "参考音频对应的准确文本。",
    "language": "Chinese",
    "response_format": "pcm",
    "stream_format": "audio",
    "seed": 1234
  }' -o outputs/online/seed.pcm
```

The response contains headerless, mono, signed 16-bit little-endian PCM at the
model's 44.1 kHz sample rate. `stream_format: audio` returns decoded audio
incrementally; `response_format: pcm` alone does not enable streaming.

## Configure latent chunks

The deployment configuration sets the transport schedule in
`connectors.connector_of_shared_memory.extra`:

```yaml
latent_chunk_size: 16
initial_latent_chunk_size: 4
latent_chunk_size_schedule: [4, 6, 8, 10, 12, 14, 16]
latent_left_context: 0
```

The schedule counts latent patches per chunk and repeats its last size until
generation ends. The final chunk flushes any remaining patches. Omitting the
schedule preserves the initial-chunk and fixed subsequent-chunk settings.
Each request advances its own schedule and retains its own VAE decoder state.

For playback, compare each chunk's arrival with the audio already received.
The decoder buffers part of the initial audio, so latent patch counts do not
directly specify the first PCM chunk's duration. Measure startup buffering and
underruns as well as first-audio latency; packet intervals alone cannot establish
continuous playback under different concurrency or network conditions.
