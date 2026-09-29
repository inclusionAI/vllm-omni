# Ming-3.0-audio-flash

> Online multilingual ASR and speech QA with optional spoken answers

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/Ming-3.0-audio-flash`
- Tasks: 16 kHz multilingual ASR and speech QA with text or 44.1 kHz audio output
- Mode: Online serving with the OpenAI-compatible Chat Completions API
- Maintainer: Community

## When to use this recipe

Use this recipe to transcribe multilingual audio or answer a question spoken in an audio clip.
ASR and Speech QA share the Ming3 audio thinker; requests for audio output additionally run the bundled Ming talker and AudioVAE.

## Serve

The default end-to-end configuration targets four 80 GB-class GPUs.
The thinker runs with TP=4 and the talker is colocated on GPU 3.

```bash
vllm serve inclusionAI/Ming-3.0-audio-flash \
    --omni \
    --deploy-config vllm_omni/deploy/ming3_audio_flash.yaml \
    --port 8091
```

`"modalities": ["text"]` stops after the thinker.
`"modalities": ["text", "audio"]` runs the talker and returns the same answer as text and speech.

### 2 x H200

For two H200 GPUs, use the validated TP=2 deployment configuration:

Example ming3_audio_flash_2gpu_h200.yaml

```yaml
base_config: ming3_audio_flash.yaml

stages:
  - stage_id: 0
    devices: "0,1"
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.88

  - stage_id: 1
    devices: "1"
    gpu_memory_utilization: 0.10
```

Launch the server with

```bash
vllm serve /path/to/inclusionAI/Ming-3.0-audio-flash \
    --omni \
    --deploy-config vllm_omni/deploy/ming3_audio_flash_2gpu_h200.yaml \
    --init-timeout 1800 \
    --stage-init-timeout 1800 \
    --port 8091
```

### Text-only deployment

When the service only needs ASR or Speech QA text, avoid loading the talker:

```bash
vllm serve inclusionAI/Ming-3.0-audio-flash \
    --omni \
    --deploy-config vllm_omni/deploy/ming3_audio_flash_thinker_only.yaml \
    --port 8091
```

## Transcribe with Chat Completions

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "/path/to/inclusionAI/Ming-3.0-audio-flash",
      "messages": [{
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Please recognize the language of this speech and transcribe it with appropriate punctuation marks, placing a `\t` between the identified language and the transcribe."
          },
          {
            "type": "audio_url",
            "audio_url": {
              "url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
            }
          }
        ]
      }],
      "modalities": ["text"],
      "temperature": 0,
      "max_tokens": 256
    }' | jq -r '.choices[0].message.content'
```

## Ask a question in audio

Example inputs:

```bash
MING3_AUDIO_FILE=/path/to/inclusionAI/Ming-3.0-audio-flash/examples/audio_files/introduce_ant_group.wav
base64 -w0 "$MING3_AUDIO_FILE" | jq -Rs '{
  model: "/path/to/inclusionAI/Ming-3.0-audio-flash",
  messages: [
    {
      role: "system",
      content: "回答音频中的问题，回答不要使用markdown格式。"
    },
    {
      role: "user",
      content: [{
        type: "audio_url",
        audio_url: {url: ("data:audio/wav;base64," + .)}
      }]
    }
  ],
  modalities: ["text"],
  temperature: 0,
  max_tokens: 1024
}' | curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    --data-binary @- | jq -r '.choices[0].message.content'
```

## Speak the answer

Using the `DB30` Talker preset (by default). The two stages stay loaded in the service.

```bash
base64 -w0 "$MING3_AUDIO_FILE" | jq -Rs '{
  model: "/path/to/inclusionAI/Ming-3.0-audio-flash",
  messages: [
    {
      role: "system",
      content: "请用一句简短的话回答音频中的问题，不超过50个汉字，不要使用Markdown格式。"
    },
    {
      role: "user",
      content: [{
        type: "audio_url",
        audio_url: {url: ("data:audio/wav;base64," + .)}
      }]
    }
  ],
  modalities: ["text", "audio"],
  additional_information: {voice_name: "DB30"},
  temperature: 0,
  max_tokens: 128
}' | curl --fail-with-body http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    --data-binary @- -o response.json
jq -r '.choices[0].message.content' response.json
jq -r '.choices[0].message.audio.data' response.json \
    | base64 -d > ming3_speech_qa.wav
```

## Notes

- The response format is `language<TAB>transcription`. Clients that need only the transcript can split the returned text at the first tab.
- Speech QA returns a direct answer rather than a transcript. It uses Chat Completions and is not passed through the ASR language-prefix postprocessor.
- The current `talker/data/voice_name.json` provides `DB30` and `lingguang`. `DB30` is the default; set `additional_information.voice_name` explicitly to select a preset, or to `null` for the zero-speaker-embedding fallback.
- The talker emits mono 44.1 kHz audio.
- Use Chat Completions for transcription; the `/v1/audio/transcriptions` endpoint is not currently available with `--omni`.
- The model handles audio chunking internally, so the deployment configuration does not set `max_audio_clip_s` or `min_energy_split_window_size`.
- Although the checkpoint advertises 131072 text positions, the provided deployment profiles cap the thinker `max_model_len` at 8192.
