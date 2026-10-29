# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask, create_sliding_window_causal_mask

from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE


class FirstChunkVAEGraph:
    @torch.inference_mode()
    def __init__(self, audio: AudioVAE, latent: torch.Tensor):
        self._latent = latent.clone()
        decoder = audio.decoder.decoder

        def empty_cache():
            cache = DynamicCache(config=decoder.config)
            # Move the cache's CPU scalar before capture.
            for layer in cache.layers:
                if hasattr(layer, "_sliding_window_tensor"):
                    layer._sliding_window_tensor = layer._sliding_window_tensor.to(latent.device)
            return cache

        def run(cache):
            stream_state = (None, None, None)
            waves = []
            for patch_idx in range(latent.shape[0]):
                wave, stream_state, cache = audio.decode(
                    self._latent[patch_idx : patch_idx + 1],
                    past_key_values=cache,
                    use_cache=True,
                    stream_state=stream_state,
                    last_chunk=False,
                )
                waves.append(wave.reshape(-1).to(torch.float32))
            return torch.cat(waves), stream_state, cache

        masks = []
        self._masks = masks  # Retain graph inputs allocated outside the capture pool.

        def record_masks(module, args, kwargs):
            inputs = kwargs["inputs_embeds"]
            cache = kwargs["past_key_values"]
            positions = (torch.arange(inputs.shape[1], device=inputs.device) + cache.get_seq_length()).unsqueeze(0)
            mask_args = dict(
                config=module.config,
                inputs_embeds=inputs,
                attention_mask=kwargs.get("attention_mask"),
                past_key_values=cache,
                position_ids=positions,
            )
            attention_mask = {"full_attention": create_causal_mask(**mask_args)}
            if module.has_sliding_layers:
                attention_mask["sliding_attention"] = create_sliding_window_causal_mask(**mask_args)
            # Precompute masks eagerly to preserve SDPA mask elision.
            masks.append(attention_mask)
            return args, dict(kwargs, attention_mask=attention_mask)

        stream = torch.cuda.Stream(device=latent.device)
        stream.wait_stream(torch.cuda.current_stream(latent.device))
        hook = decoder.register_forward_pre_hook(record_masks, with_kwargs=True)
        try:
            with torch.cuda.stream(stream):
                run(empty_cache())
        finally:
            hook.remove()
        torch.cuda.current_stream(latent.device).wait_stream(stream)

        mask_iter = iter(masks)

        def supply_masks(module, args, kwargs):
            return args, dict(kwargs, attention_mask=next(mask_iter))

        cache = empty_cache()
        self._graph = torch.cuda.CUDAGraph()
        hook = decoder.register_forward_pre_hook(supply_masks, with_kwargs=True)
        try:
            with torch.cuda.graph(self._graph, stream=stream):
                self._output = run(cache)
        finally:
            hook.remove()

    @torch.inference_mode()
    def replay(self, latent: torch.Tensor):
        if (
            latent.shape != self._latent.shape
            or latent.dtype != self._latent.dtype
            or latent.device != self._latent.device
        ):
            raise ValueError("LingV3 VAE graph inputs must match the captured shape, dtype and device")
        self._latent.copy_(latent)
        self._graph.replay()
        # Isolate each request's waveform and cache from the next replay.
        return copy.deepcopy(self._output)
