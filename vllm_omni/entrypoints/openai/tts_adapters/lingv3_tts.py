# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI speech adapter for inclusionAI LingV3-TTS."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoTokenizer
from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.lingv3_tts.constants import (
    AUDIO_END_TOKEN_ID,
    MAX_DECODE_STEPS,
)
from vllm_omni.model_executor.models.lingv3_tts.prompt_assembly import (
    build_lingv3_prompt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class LingV3TTSAdapter(ARTTSAdapter):
    name = "lingv3_tts"
    model_archs = frozenset({"LingV3TTSForConditionalGeneration"})
    arch_identifies_entry_stage = True
    detect_priority = 190
    max_new_tokens_max = MAX_DECODE_STEPS

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio is None:
            return "LingV3-TTS requires 'ref_audio'"
        if isinstance(request.ref_audio, list):
            return "LingV3-TTS v1 accepts exactly one 'ref_audio' clip"
        format_error = self.ctx.server._validate_ref_audio_format(request.ref_audio)
        if format_error:
            return format_error
        if request.ref_text is not None and not request.ref_text.strip():
            return "'ref_text' must be non-empty when provided"
        if request.instructions:
            return "LingV3-TTS v1 does not support 'instructions'"
        voice = request.voice.lower() if isinstance(request.voice, str) else None
        if voice and voice in getattr(self.ctx.server, "uploaded_speakers", {}):
            return "LingV3-TTS v1 does not support stored voice presets"
        if request.speaker_embedding is not None:
            return "LingV3-TTS v1 derives speaker conditioning from 'ref_audio'"
        if request.max_new_tokens is not None:
            if request.max_new_tokens < 5:
                return "max_new_tokens must be at least 5"
            if request.max_new_tokens > MAX_DECODE_STEPS:
                return f"max_new_tokens cannot exceed {MAX_DECODE_STEPS}"
        return None

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del sampling_params_list, has_inline_ref_audio
        server = self.ctx.server
        if server._tts_tokenizer is None:
            model_name = server.engine_client.model_config.model
            trust_remote_code = bool(getattr(server.engine_client.model_config, "trust_remote_code", False))
            server._tts_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        samples, sample_rate = await server._resolve_ref_audio(request.ref_audio)
        prompt_dict = build_lingv3_prompt(
            server._tts_tokenizer,
            target_text=request.input,
            reference_waveform=torch.as_tensor(samples, dtype=torch.float32).reshape(1, -1),
            reference_sample_rate=int(sample_rate),
            prompt_text=request.ref_text,
            max_decode_steps=request.max_new_tokens,
        )
        prompt = tokens_input(prompt_token_ids=prompt_dict["prompt_token_ids"])
        prompt["prompt"] = prompt_dict["prompt"]
        prompt["text"] = prompt_dict["text"]
        prompt["additional_information"] = prompt_dict["additional_information"]
        return PreparedRequest(
            prompt=prompt,
            tts_params=prompt_dict["additional_information"],
            model_type=self.name,
        )

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        del prompt, request_id
        updated = copy.deepcopy(sampling_params_list)
        updated[0].stop_token_ids = [AUDIO_END_TOKEN_ID]
        latent_cap = int(request.max_new_tokens or MAX_DECODE_STEPS)
        updated[0].max_tokens = latent_cap + 1
        return updated


__all__ = ["LingV3TTSAdapter"]
