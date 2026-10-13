# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
from typing import Any

import torch

from vllm_omni.model_executor.models.common.ming.speaker_extractor import (
    SpeakerEmbeddingExtractor,
)
from vllm_omni.model_executor.models.ming_tts.audio_prep import (
    coerce_prompt_waveform,
    coerce_speaker_embeddings,
)

from .constants import (
    KEY_SPEAKER_EMBEDDING,
    KEY_SPEAKER_SAMPLE_RATES,
    KEY_SPEAKER_WAVEFORM,
    KEY_SPEAKER_WAVEFORM_LENGTHS,
)

_LOAD_LOCK = threading.Lock()
# TODO: Move shared lazy CAM++ loading and locking into common.ming.speaker_extractor.


def resolve_speaker_embeddings(wrapper: Any, info: dict[str, Any]) -> list[torch.Tensor]:
    direct = info.get(KEY_SPEAKER_EMBEDDING)
    if direct is not None:
        embeddings = coerce_speaker_embeddings(direct)
        if embeddings:
            return embeddings

    waveform = info.get(KEY_SPEAKER_WAVEFORM)
    if waveform is None:
        raise ValueError("LingV3-TTS requires reference audio for speaker conditioning")
    waveform = coerce_prompt_waveform(waveform)
    lengths = torch.as_tensor(info.get(KEY_SPEAKER_WAVEFORM_LENGTHS, [waveform.shape[-1]])).reshape(-1)
    sample_rates = torch.as_tensor(info.get(KEY_SPEAKER_SAMPLE_RATES, [wrapper.ming_config.sample_rate])).reshape(-1)
    if lengths.numel() != 1 or sample_rates.numel() != 1:
        raise ValueError("LingV3-TTS v1 accepts exactly one reference audio clip")

    extractor = getattr(wrapper, "_speaker_extractor", None)
    if extractor is None:
        with _LOAD_LOCK:
            extractor = getattr(wrapper, "_speaker_extractor", None)
            if extractor is None:
                extractor = SpeakerEmbeddingExtractor(wrapper.vllm_config.model_config.model, target_sr=16000)
                wrapper._speaker_extractor = extractor
    embedding = extractor.extract_from_waveform(waveform[:, : int(lengths[0])], int(sample_rates[0]))
    embeddings = coerce_speaker_embeddings(embedding)
    if not embeddings:
        raise RuntimeError("CAM++ returned no LingV3 speaker embedding")
    return embeddings


__all__ = ["resolve_speaker_embeddings"]
