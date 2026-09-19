# Ming-3.0-audio-flash

> Online serving for multilingual ASR and single-turn speech QA

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/Ming-3.0-audio-flash`
- Tasks: 16 kHz multilingual ASR and audio question answering with text output
- Mode: Online serving with the OpenAI-compatible Chat Completions API
- Maintainer: Community

## When to use this recipe

Use this recipe to transcribe multilingual audio or answer a question spoken
in an audio clip. Both tasks use the same audio-understanding stage; the ChatML
messages select the behavior.

## Serve

The bundled deployment configuration targets four 80 GB-class GPUs with
tensor parallelism, chunked prefill, and eager execution.

```bash
vllm serve inclusionAI/Ming-3.0-audio-flash \
    --omni \
    --deploy-config vllm_omni/deploy/ming3_audio_flash.yaml \
    --port 8091
```

### 2 x H200 profile

For two H200 GPUs, use the validated TP=2 deployment configuration:

Example ming3_audio_flash_2gpu_h200.yaml

```yaml
base_config: ming3_audio_flash.yaml

stages:
  - stage_id: 0
    devices: "0,1"
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.95
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

The following request mirrors the vendor Speech QA example. Replace the local
path if the checkpoint is stored elsewhere.

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

## Notes

- The response format is `language<TAB>transcription`. Clients that need only
  the transcript can split the returned text at the first tab.
- Speech QA returns a direct answer rather than a transcript. It uses Chat
  Completions and is not passed through the ASR language-prefix postprocessor.
- Use Chat Completions for transcription; the `/v1/audio/transcriptions`
  endpoint is not currently available with `--omni`.
- The model handles audio chunking internally, so the deployment configuration
  does not set `max_audio_clip_s` or `min_energy_split_window_size`.
