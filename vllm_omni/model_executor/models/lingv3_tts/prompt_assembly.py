# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact legacy-antmmf prompt contract used to train LingV3-TTS."""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.model_executor.models.ming_tts.audio_prep import (
    _normalize_prompt_waveform,
    count_prompt_waveform_patches,
    pad_prompt_waveform,
)

from .constants import (
    AUDIO_END_TOKEN_ID,
    AUDIO_PATCH_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    KEY_MAX_DECODE_STEPS,
    KEY_SPEAKER_EMBEDDING,
    KEY_SPEAKER_SAMPLE_RATES,
    KEY_SPEAKER_WAVEFORM,
    KEY_SPEAKER_WAVEFORM_LENGTHS,
    MAX_DECODE_STEPS,
    SPEAKER_END_TOKEN_ID,
    SPEAKER_START_TOKEN_ID,
    TEXT_EOS_TOKEN_ID,
)

USER_PREFIX = "<role>HUMAN</role>Please generate speech based on the following description.\n  speaker_1:"
TEXT_PREFIX = "\n Text input:\n"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"
EXPECTED_TOKENS = {
    "<audioPatch>": AUDIO_PATCH_TOKEN_ID,
    "<audio>": AUDIO_START_TOKEN_ID,
    "</audio>": AUDIO_END_TOKEN_ID,
    "<spk>": SPEAKER_START_TOKEN_ID,
    "</spk>": SPEAKER_END_TOKEN_ID,
}


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def validate_lingv3_tokenizer(tokenizer: Any) -> dict[str, int]:
    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in EXPECTED_TOKENS}
    mismatches = {
        token: (expected, actual[token]) for token, expected in EXPECTED_TOKENS.items() if actual[token] != expected
    }
    if mismatches:
        details = ", ".join(f"{token}: expected {want}, got {got}" for token, (want, got) in mismatches.items())
        raise ValueError("Tokenizer IDs do not match LingV3-TTS training: " + details)
    for token, token_id in EXPECTED_TOKENS.items():
        encoded = _encode(tokenizer, token)
        if encoded != [token_id]:
            raise ValueError(f"{token!r} must encode to [{token_id}], got {encoded}")
    if int(tokenizer.eos_token_id) != TEXT_EOS_TOKEN_ID:
        raise ValueError(f"Tokenizer EOS must be {TEXT_EOS_TOKEN_ID}, got {tokenizer.eos_token_id}")
    return actual


def build_lingv3_prompt_token_ids(
    tokenizer: Any,
    *,
    target_text: str,
    prompt_text: str | None = None,
    prompt_latent_steps: int = 0,
) -> tuple[list[int], int, int | None]:
    if not target_text or not target_text.strip():
        raise ValueError("target_text must not be empty")
    if prompt_text is not None and not prompt_text.strip():
        raise ValueError("prompt_text must be non-empty when provided")
    if prompt_latent_steps and prompt_text is None:
        raise ValueError("prompt_text is required when prompt latents are present")
    validate_lingv3_tokenizer(tokenizer)
    ids = _encode(tokenizer, USER_PREFIX)
    ids.append(SPEAKER_START_TOKEN_ID)
    speaker_position = len(ids)
    ids.append(AUDIO_PATCH_TOKEN_ID)
    ids.append(SPEAKER_END_TOKEN_ID)
    ids.extend(_encode(tokenizer, TEXT_PREFIX))
    if prompt_text is not None:
        ids.extend(_encode(tokenizer, prompt_text))
    ids.extend(_encode(tokenizer, target_text))
    ids.extend(_encode(tokenizer, ASSISTANT_PREFIX))
    ids.append(AUDIO_START_TOKEN_ID)
    prompt_position = len(ids) if prompt_latent_steps else None
    ids.extend([AUDIO_PATCH_TOKEN_ID] * int(prompt_latent_steps))
    return ids, speaker_position, prompt_position


def build_lingv3_prompt(
    tokenizer: Any,
    *,
    target_text: str,
    reference_waveform: torch.Tensor,
    reference_sample_rate: int,
    prompt_text: str | None = None,
    speaker_embedding: Any = None,
    max_decode_steps: int | None = None,
) -> dict[str, Any]:
    if reference_waveform is None:
        raise ValueError("LingV3-TTS requires reference audio")
    prompt_waveform = None
    prompt_steps = 0
    if prompt_text is not None:
        prompt_waveform = pad_prompt_waveform(
            _normalize_prompt_waveform((reference_waveform, reference_sample_rate), target_sr=44100),
            patch_size=1,
            sample_rate=44100,
        )
        prompt_steps = count_prompt_waveform_patches(
            prompt_waveform,
            patch_size=1,
            frame_hop=882,
            vae_patch_size=4,
        )
    ids, _, _ = build_lingv3_prompt_token_ids(
        tokenizer,
        target_text=target_text,
        prompt_text=prompt_text,
        prompt_latent_steps=prompt_steps,
    )
    info: dict[str, Any] = {
        KEY_SPEAKER_WAVEFORM: reference_waveform,
        KEY_SPEAKER_WAVEFORM_LENGTHS: torch.tensor([reference_waveform.shape[-1]], dtype=torch.int32),
        KEY_SPEAKER_SAMPLE_RATES: torch.tensor([reference_sample_rate], dtype=torch.int32),
        KEY_MAX_DECODE_STEPS: int(max_decode_steps or MAX_DECODE_STEPS),
    }
    if speaker_embedding is not None:
        info[KEY_SPEAKER_EMBEDDING] = speaker_embedding
    if prompt_waveform is not None:
        info["prompt_waveform"] = prompt_waveform
    return {
        "prompt_token_ids": ids,
        "prompt": USER_PREFIX,
        "text": target_text,
        "additional_information": info,
    }


__all__ = [
    "ASSISTANT_PREFIX",
    "TEXT_PREFIX",
    "USER_PREFIX",
    "build_lingv3_prompt",
    "build_lingv3_prompt_token_ids",
    "validate_lingv3_tokenizer",
]
