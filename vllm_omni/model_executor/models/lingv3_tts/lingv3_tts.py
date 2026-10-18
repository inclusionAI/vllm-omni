# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-stage vLLM-Omni wrapper for LingV3-TTS."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.ming_tts import config_ming_tts as state_keys
from vllm_omni.model_executor.models.ming_tts.audio_prep import (
    _find_audio_placeholder_positions,
    _find_speaker_placeholder_positions,
    _initial_history,
    _take_scalar,
)
from vllm_omni.model_executor.models.ming_tts.ming_tts import (
    _ModelSampleAdapter,
)
from vllm_omni.model_executor.models.ming_tts.prompt_encoder import _resolve_prompt_latents

from .config_lingv3_tts import LingV3TTSRuntimeConfig, get_lingv3_backbone_config
from .constants import (
    KEY_CFG,
    KEY_LAST_STOP_PROB,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_SPEAKER_EMBEDDING,
    KEY_SPEAKER_WAVEFORM,
    KEY_STOP_REASON,
    KEY_TEMPERATURE,
)
from .speaker_extractor import resolve_speaker_embeddings
from .weight_accounting import validate_lingv3_checkpoint_keys


class LingV3TTSForConditionalGeneration(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsPP,
    CustomProcessMixin,
):
    packed_modules_mapping = BailingMoeV3ForCausalLM.packed_modules_mapping
    requires_request_sample_eligibility = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.ming_config = LingV3TTSRuntimeConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self.lingv3_config = self.ming_config
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.requires_raw_input_tokens = False
        self.model_stage = vllm_config.model_config.model_stage
        self._prompt_encoder = None
        self._speaker_extractor = None

        if self.model_stage == "llm":
            from vllm_omni.utils.speaker_cache import get_speaker_cache

            self._speaker_cache = get_speaker_cache()
            self.model = init_vllm_registered_model(
                vllm_config=vllm_config,
                architectures=["LingV3TTSLLMModel"],
            )
            self.has_preprocess = True
            self.has_postprocess = True
            self.set_custom_preprocess(self.preprocess)
            self.set_custom_postprocess(self.postprocess)
        elif self.model_stage == "audio_vae":
            self.model = init_vllm_registered_model(
                vllm_config=vllm_config,
                architectures=["LingV3TTSAudioVAEModel"],
            )
            self.requires_raw_input_tokens = True
        else:
            raise ValueError(f"Invalid LingV3 model_stage={self.model_stage!r}")
        self.make_empty_intermediate_tensors = getattr(self.model, "make_empty_intermediate_tensors", lambda: None)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return BailingMoeV3ForCausalLM.get_mamba_state_shape_from_config(get_lingv3_backbone_config(vllm_config))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return BailingMoeV3ForCausalLM.get_mamba_state_dtype_from_config(get_lingv3_backbone_config(vllm_config))

    @classmethod
    def get_mamba_state_copy_func(cls):
        return BailingMoeV3ForCausalLM.get_mamba_state_copy_func()

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sample"):
            return _ModelSampleAdapter(self.model)
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any):
        return self.model.embed_input_ids(input_ids=input_ids, **kwargs)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return self.model.compute_logits(hidden_states, sampling_metadata=sampling_metadata)

    def sample(self, logits, sampling_metadata):
        return self.model.sample(logits, sampling_metadata)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        callback = getattr(self.model, "on_requests_finished", None)
        if callback is not None:
            callback(finished_req_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        validate_lingv3_checkpoint_keys(name for name, _ in weights)
        allowed = (
            ("model.", "linear_proj_audio.", "flowloss.", "stop_head.", "spk_head.")
            if self.model_stage == "llm"
            else ("audio.decoder.",)
        )
        selected = [(name, value) for name, value in weights if name.startswith(allowed)]
        if not selected:
            raise RuntimeError(f"LingV3 {self.model_stage} stage received no weights with prefixes {allowed}")
        loaded = self.model.load_weights(selected)
        return {f"model.{name}" for name in loaded}

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info: Any,
    ):
        if self.model_stage != "llm":
            return input_ids, input_embeds, {}
        input_embeds = self.model.embed_input_ids(input_ids).clone()
        return (
            self._prefill_preprocess(input_ids, input_embeds, info)
            if input_ids.shape[0] > 1
            else self._decode_preprocess(input_ids, input_embeds, info)
        )

    def _prefill_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        info: dict[str, Any],
    ):
        mute_speaker = bool(_take_scalar(info.get("mute_speaker"), 0))
        no_speaker_slot = bool(_take_scalar(info.get("no_speaker_slot"), 0))
        if mute_speaker and no_speaker_slot:
            raise ValueError("Speaker slot cannot be both absent and muted")
        if no_speaker_slot and any(
            info.get(key) is not None for key in (KEY_SPEAKER_EMBEDDING, KEY_SPEAKER_WAVEFORM, "prompt_waveform")
        ):
            raise ValueError("No-speaker-slot conditioning cannot include speaker or prompt audio")
        prompt_generator = None
        if info.get("prompt_waveform") is not None:
            prompt_generator = self.model._get_request_generator(info, input_embeds.device)
        prompt_latents = (
            None
            if mute_speaker or no_speaker_slot
            else _resolve_prompt_latents(
                self, info, generator=prompt_generator, posterior_parameterization="softplus_scale"
            )
        )
        history = _initial_history(
            prompt_latents["frames"] if prompt_latents is not None else None,
            history_size=self.ming_config.history_patch_size,
            latent_dim=self.ming_config.latent_dim,
            device=input_embeds.device,
            dtype=torch.float32,
        )
        update: dict[str, Any] = {
            state_keys.KEY_LATENT_HISTORY: history.detach().cpu().contiguous(),
            state_keys.KEY_DECODE_STEP: 0,
        }

        slots = _find_speaker_placeholder_positions(input_ids, self.ming_config)
        if no_speaker_slot:
            if slots:
                raise RuntimeError("No-speaker-slot conditioning must not contain speaker slots")
        elif mute_speaker:
            if len(slots) != 1:
                raise RuntimeError("Masked speaker conditioning requires exactly one speaker slot")
            input_embeds[slots[0]].zero_()
        else:
            speakers = resolve_speaker_embeddings(self, info)
            if not slots or len(slots) != len(speakers):
                raise RuntimeError(f"LingV3 speaker slot/embedding count mismatch: {len(slots)}/{len(speakers)}")
            for slot, speaker in zip(slots, speakers, strict=True):
                input_embeds[slot] = self.model.project_speaker_embedding(
                    speaker.to(input_embeds.device, input_embeds.dtype).unsqueeze(0)
                ).squeeze(0)

        if prompt_latents is not None:
            patches = prompt_latents["patches"].to(device=input_embeds.device, dtype=self.model.fm_dtype)
            prompt_embeds = self.model.linear_proj_audio(patches).squeeze(1)
            positions = _find_audio_placeholder_positions(input_ids, self.ming_config)
            if positions.numel() != prompt_embeds.shape[0]:
                raise RuntimeError(
                    "LingV3 prompt latent/token count mismatch: "
                    f"{prompt_embeds.shape[0]} latents for {positions.numel()} slots"
                )
            input_embeds[positions] = prompt_embeds.to(input_embeds.dtype)

        self._copy_request_state(update, info)
        return input_ids, input_embeds, update

    def _decode_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        info: dict[str, Any],
    ):
        history = info.get(state_keys.KEY_LATENT_HISTORY)
        if not isinstance(history, torch.Tensor):
            history = torch.zeros(
                (self.ming_config.history_patch_size, self.ming_config.latent_dim),
                dtype=torch.float32,
            )
        update: dict[str, Any] = {
            state_keys.KEY_LATENT_HISTORY: history.detach().cpu().contiguous(),
            state_keys.KEY_DECODE_STEP: int(info.get(state_keys.KEY_DECODE_STEP, 0)),
        }
        next_embeds = info.get(state_keys.KEY_NEXT_EMBEDS)
        if isinstance(next_embeds, torch.Tensor):
            input_embeds[0] = next_embeds.reshape(-1, self.ming_config.llm_hidden_size)[0].to(
                input_embeds.device, input_embeds.dtype
            )
        self._copy_request_state(update, info)
        return input_ids, input_embeds, update

    def _copy_request_state(self, update: dict[str, Any], info: dict[str, Any]) -> None:
        request_id = info.get(state_keys.KEY_REQUEST_ID, info.get(KEY_REQUEST_ID, info.get("request_id")))
        if request_id is not None:
            update[state_keys.KEY_REQUEST_ID] = request_id
        if "tts_local_seed" in info:
            update["tts_local_seed"] = info["tts_local_seed"]
        mappings = (
            (KEY_CFG, state_keys.KEY_CFG),
            (KEY_SIGMA, state_keys.KEY_SIGMA),
            (KEY_TEMPERATURE, state_keys.KEY_TEMPERATURE),
            (KEY_MAX_DECODE_STEPS, state_keys.KEY_MAX_DECODE_STEPS),
            (KEY_MIN_DECODE_STEPS, state_keys.KEY_MIN_DECODE_STEPS),
        )
        for public_key, state_key in mappings:
            if state_key in info:
                update[state_key] = info[state_key]
            elif public_key in info:
                update[state_key] = info[public_key]

    def postprocess(self, hidden_states: torch.Tensor, **info: Any) -> dict[str, Any]:
        if self.model_stage != "llm" or hidden_states.numel() == 0:
            return {}
        request_id = info.get(state_keys.KEY_REQUEST_ID, info.get("request_id"))
        pending = self.model.pop_postprocess_update(request_id)
        if not pending or not isinstance(pending.get(self.model.latent_patch_key), torch.Tensor):
            return {}
        update = {
            state_keys.KEY_LATENT_HISTORY: pending[state_keys.KEY_LATENT_HISTORY].detach().cpu().contiguous(),
            state_keys.KEY_NEXT_EMBEDS: pending[state_keys.KEY_NEXT_EMBEDS].detach().cpu().contiguous(),
            state_keys.KEY_DECODE_STEP: int(info.get(state_keys.KEY_DECODE_STEP, 0)) + 1,
        }
        if "tts_local_seed" in info:
            update["tts_local_seed"] = info["tts_local_seed"]
        stop_prob = _take_scalar(pending.get(self.model.stop_probability_key), 0)
        if stop_prob is not None:
            update[KEY_LAST_STOP_PROB] = stop_prob
        stop_reason = pending.get(KEY_STOP_REASON)
        if isinstance(stop_reason, str):
            update[KEY_STOP_REASON] = stop_reason
        if request_id is not None:
            update[state_keys.KEY_REQUEST_ID] = request_id
        return update


__all__ = ["LingV3TTSForConditionalGeneration"]
