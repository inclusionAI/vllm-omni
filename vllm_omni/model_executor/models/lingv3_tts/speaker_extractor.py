# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
    if lengths.numel() == 0 or lengths.numel() != sample_rates.numel():
        raise ValueError("Each LingV3 reference audio clip requires a length and sample rate")
    if (lengths <= 0).any() or (sample_rates <= 0).any() or int(lengths.sum()) != waveform.shape[-1]:
        raise ValueError("Invalid LingV3 reference audio lengths or sample rates")
    embeddings = []
    offset = 0
    for length, sample_rate in zip(lengths.tolist(), sample_rates.tolist(), strict=True):
        extractor = getattr(wrapper, "_speaker_extractor", None)
        if extractor is None:
            with _LOAD_LOCK:
                extractor = getattr(wrapper, "_speaker_extractor", None)
                if extractor is None:
                    extractor = SpeakerEmbeddingExtractor(wrapper.vllm_config.model_config.model, target_sr=16000)
                    wrapper._speaker_extractor = extractor
        embedding = extractor.extract_from_waveform(waveform[:, offset : offset + length], sample_rate)
        extracted = coerce_speaker_embeddings(embedding)
        if len(extracted) != 1:
            raise RuntimeError("CAM++ must return one LingV3 speaker embedding per reference clip")
        embeddings.extend(extracted)
        offset += length
    return embeddings


__all__ = ["resolve_speaker_embeddings"]
