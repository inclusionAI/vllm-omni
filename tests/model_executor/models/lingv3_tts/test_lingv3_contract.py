# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.common.ming.cfm_cudagraph import CFMGraphExecutor
from vllm_omni.model_executor.models.common.ming.continuous_tts import (
    STOP_REASON_CONTINUE,
    STOP_REASON_MAX_DECODE_STEPS,
    STOP_REASON_STOP_HEAD,
    resolve_stop_decision,
)
from vllm_omni.model_executor.models.lingv3_tts.config_lingv3_tts import (
    LingV3TTSConfig,
    LingV3TTSRuntimeConfig,
)
from vllm_omni.model_executor.models.lingv3_tts.constants import (
    AUDIO_END_TOKEN_ID,
    AUDIO_PATCH_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    SPEAKER_END_TOKEN_ID,
    SPEAKER_START_TOKEN_ID,
)
from vllm_omni.model_executor.models.lingv3_tts.lingv3_tts import (
    LingV3TTSForConditionalGeneration,
)
from vllm_omni.model_executor.models.lingv3_tts.lingv3_tts_llm import (
    LingV3TTSLLMModel,
)
from vllm_omni.model_executor.models.lingv3_tts.prompt_assembly import (
    ASSISTANT_PREFIX,
    TEXT_PREFIX,
    USER_PREFIX,
    build_lingv3_prompt,
    build_lingv3_prompt_token_ids,
    validate_lingv3_tokenizer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class FakeTokenizer:
    eos_token_id = 156892
    special = {
        "<audioPatch>": AUDIO_PATCH_TOKEN_ID,
        "<audio>": AUDIO_START_TOKEN_ID,
        "</audio>": AUDIO_END_TOKEN_ID,
        "<spk>": SPEAKER_START_TOKEN_ID,
        "</spk>": SPEAKER_END_TOKEN_ID,
    }

    def convert_tokens_to_ids(self, token):
        return self.special.get(token)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self.special:
            return [self.special[text]]
        return [1000 + ord(char) for char in text]


def _config() -> LingV3TTSConfig:
    backbone = {
        "model_type": "bailing_moe_v3",
        "vocab_size": 157184,
        "hidden_size": 1536,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "layer_group_size": 4,
        "head_dim": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "short_conv_kernel_size": 4,
        "kda_safe_gate": True,
        "kda_lower_bound": -5.0,
    }
    audio = {
        "sample_rate": 44100,
        "patch_size": 4,
        "semantic_module_kwargs": None,
        "enc_kwargs": {"hop_size": 882, "latent_dim": 64},
        "dec_kwargs": {"output_dim": 882, "latent_dim": 64},
    }
    return LingV3TTSConfig(
        llm_config=backbone,
        audio_config=audio,
        tts_config={
            "sample_rate": 44100,
            "latent_dim": 64,
            "speaker_embedding_dim": 192,
            "patch_size": 1,
            "history_size": 1,
            "audio_patch_token_id": AUDIO_PATCH_TOKEN_ID,
            "audio_start_token_id": AUDIO_START_TOKEN_ID,
            "audio_end_token_id": AUDIO_END_TOKEN_ID,
            "speaker_start_token_id": SPEAKER_START_TOKEN_ID,
            "speaker_end_token_id": SPEAKER_END_TOKEN_ID,
        },
        aggregator_config={"in_channels": 64, "hidden_size": 1024},
        flow_config={"z_channels": 64, "hidden_size": 1024},
    )


def test_config_normalizes_bailing_and_keeps_independent_geometry():
    config = _config()
    runtime = LingV3TTSRuntimeConfig.from_hf_config(config)
    runtime.validate()
    assert config.llm_config.model_type == "bailing_hybrid"
    assert config.llm_config.checkpoint_model_type == "bailing_moe_v3"
    assert config.llm_config.kda_safe_gate is False
    assert config.llm_config.kda_lower_bound is None
    assert config.llm_config.kv_lora_rank + config.llm_config.qk_rope_head_dim == 576
    assert runtime.patch_size == runtime.history_patch_size == 1
    assert runtime.vae_patch_size == 4


def test_prompt_contract_speaker_only():
    tokenizer = FakeTokenizer()
    ids, speaker_position, prompt_position = build_lingv3_prompt_token_ids(tokenizer, target_text="target")
    expected = (
        tokenizer.encode(USER_PREFIX)
        + [SPEAKER_START_TOKEN_ID, AUDIO_PATCH_TOKEN_ID, SPEAKER_END_TOKEN_ID]
        + tokenizer.encode(TEXT_PREFIX)
        + tokenizer.encode("target")
        + tokenizer.encode(ASSISTANT_PREFIX)
        + [AUDIO_START_TOKEN_ID]
    )
    assert ids == expected
    assert ids[speaker_position] == AUDIO_PATCH_TOKEN_ID
    assert prompt_position is None


def test_prompt_contract_requires_reference_audio():
    with pytest.raises(ValueError, match="requires reference audio"):
        build_lingv3_prompt(
            FakeTokenizer(),
            target_text="target",
            reference_waveform=None,
            reference_sample_rate=24000,
        )


def test_prompt_contract_concatenates_reference_and_target_without_separator():
    tokenizer = FakeTokenizer()
    ids, _, prompt_position = build_lingv3_prompt_token_ids(
        tokenizer,
        target_text="target",
        prompt_text="reference",
        prompt_latent_steps=3,
    )
    joined = tokenizer.encode("reference") + tokenizer.encode("target")
    text_start = len(tokenizer.encode(USER_PREFIX)) + 3 + len(tokenizer.encode(TEXT_PREFIX))
    assert ids[text_start : text_start + len(joined)] == joined
    assert ids[prompt_position:] == [AUDIO_PATCH_TOKEN_ID] * 3


def test_prefill_uses_vendor_prompt_posterior_and_continues_request_rng(monkeypatch):
    generator = torch.Generator().manual_seed(17)
    calls: dict[str, Any] = {}

    def get_request_generator(info, device):
        calls["generator_request"] = (info, device)
        return generator

    def resolve_prompt_latents(wrapper, info, **kwargs):
        calls["prompt"] = (wrapper, info, kwargs)
        return None

    from vllm_omni.model_executor.models.lingv3_tts import lingv3_tts as model_module

    monkeypatch.setattr(model_module, "_resolve_prompt_latents", resolve_prompt_latents)
    monkeypatch.setattr(model_module, "resolve_speaker_embeddings", lambda wrapper, info: [torch.zeros(192)])
    monkeypatch.setattr(model_module, "_find_speaker_placeholder_positions", lambda input_ids, config: [0])

    model = SimpleNamespace(
        _get_request_generator=get_request_generator,
        fm_dtype=torch.float32,
        project_speaker_embedding=lambda embedding: torch.zeros((1, 4)),
    )
    wrapper: Any = SimpleNamespace(
        model=model,
        ming_config=SimpleNamespace(history_patch_size=1, latent_dim=64),
        _copy_request_state=lambda update, info: None,
    )
    info = {"prompt_waveform": torch.ones((1, 8)), "request_id": "request-1"}

    LingV3TTSForConditionalGeneration._prefill_preprocess(
        wrapper,
        torch.tensor([AUDIO_PATCH_TOKEN_ID]),
        torch.zeros((1, 4)),
        info,
    )

    assert calls["generator_request"] == (info, torch.device("cpu"))
    assert calls["prompt"][2] == {
        "generator": generator,
        "posterior_parameterization": "clamped_logvar",
    }


@pytest.mark.parametrize(
    ("info", "expected_seed"),
    [
        pytest.param({"request_id": "request-1", "tts_local_seed": 1234}, 1234, id="explicit-seed"),
        pytest.param({"request_id": "request-1"}, None, id="unseeded-speaker"),
    ],
)
def test_request_generator_seed_policy(info, expected_seed):
    model = SimpleNamespace(_request_generators={})
    generator = LingV3TTSLLMModel._get_request_generator(
        model,
        info,
        torch.device("cpu"),
    )

    actual_seed = None if generator is None else generator.initial_seed()
    assert actual_seed == expected_seed


def test_tokenizer_mismatch_is_not_repaired():
    tokenizer = FakeTokenizer()
    tokenizer.special["<audio>"] = -1
    with pytest.raises(ValueError, match="expected 157165"):
        validate_lingv3_tokenizer(tokenizer)


@pytest.mark.parametrize(
    ("emitted", "probability", "reason", "token"),
    [
        (4, 1.0, STOP_REASON_CONTINUE, AUDIO_PATCH_TOKEN_ID),
        (5, 0.51, STOP_REASON_STOP_HEAD, AUDIO_END_TOKEN_ID),
        (800, 0.0, STOP_REASON_MAX_DECODE_STEPS, AUDIO_END_TOKEN_ID),
    ],
)
def test_stop_is_evaluated_after_emitting_latent(emitted, probability, reason, token):
    decision = resolve_stop_decision(
        emitted_steps=emitted,
        stop_probability=probability,
        stop_threshold=0.5,
        minimum_steps=5,
        maximum_steps=800,
        continue_token_id=AUDIO_PATCH_TOKEN_ID,
        stop_token_id=AUDIO_END_TOKEN_ID,
    )
    assert decision.reason == reason
    assert decision.next_token_id == token


def test_runtime_rejects_flow_vae_patch_confusion():
    config = _config()
    config.audio_config.patch_size = 1
    runtime = LingV3TTSRuntimeConfig.from_hf_config(config)
    with pytest.raises(ValueError, match="AudioVAE patch_size must be 4"):
        runtime.validate()


def test_runtime_rejects_incompatible_mla_geometry():
    config = _config()
    config.llm_config.kv_lora_rank = 256
    runtime = LingV3TTSRuntimeConfig.from_hf_config(config)
    with pytest.raises(ValueError, match="expected MLA head size 576"):
        runtime.validate()


def test_pipeline_contract():
    from vllm_omni.model_executor.models.lingv3_tts.pipeline import (
        LINGV3_TTS_PIPELINE,
    )

    stage0, stage1 = LINGV3_TTS_PIPELINE.stages
    assert stage0.hf_config_name == "llm_config"
    assert stage0.owns_tokenizer is True
    assert stage0.sampling_constraints["max_tokens"] == 801
    assert stage1.final_output is True
    assert stage1.final_output_type == "audio"


def test_cfm_graph_bucket_zero_pads_without_touching_live_rows():
    executor = SimpleNamespace(batch_size=4)
    live = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    solver_noise = live.unsqueeze(0).expand(10, -1, -1)

    model = SimpleNamespace(_cfm_graph_batch_sizes=[1, 2, 4])
    assert LingV3TTSLLMModel._select_cfm_graph_batch_size(model, 3) == 4
    padded = CFMGraphExecutor._pad_batch(executor, live)
    padded_solver_noise = CFMGraphExecutor._pad_batch(executor, solver_noise, dim=1)

    assert padded.shape == (4, 2)
    assert padded_solver_noise.shape == (10, 4, 2)
    torch.testing.assert_close(padded[:3], live)
    torch.testing.assert_close(padded_solver_noise[:, :3], solver_noise)
    assert torch.count_nonzero(padded[3]) == 0
    assert torch.count_nonzero(padded_solver_noise[:, 3]) == 0


def test_lingv3_vectorizes_only_compatible_flow_requests():
    config = SimpleNamespace(
        history_patch_size=2,
        latent_dim=3,
        patch_size=1,
        llm_hidden_size=4,
        cfg=2.0,
        sigma=0.25,
        temperature=0.0,
    )
    calls = []

    def get_request_generator(info, device):
        return torch.Generator(device=device).manual_seed(info["seed"])

    def decode_one_step(**kwargs):
        calls.append(kwargs)
        batch_size = kwargs["hidden_states"].shape[0]
        sampled = kwargs["initial_noise"].clone()
        next_embeds = torch.zeros((batch_size, 1, config.llm_hidden_size))
        history = torch.cat([kwargs["latent_history"][:, config.patch_size :], sampled], dim=1)
        stop_probs = torch.zeros(batch_size)
        return sampled, next_embeds, history, stop_probs

    model = SimpleNamespace(
        vectorize_cfm_decode=True,
        _cfm_graph_enabled=True,
        fm_dtype=torch.float32,
        ming_config=config,
        _get_request_generator=get_request_generator,
        _decode_one_step=decode_one_step,
    )
    model._draw_request_cfm_noises = lambda generators, device: (
        LingV3TTSLLMModel._draw_request_cfm_noises(model, generators, device)
    )
    history = torch.zeros((1, config.history_patch_size, config.latent_dim))
    request_infos = [
        {"ming_latent_history": history, "ming_cfg": 2.0, "seed": 11},
        {"ming_latent_history": history, "ming_cfg": 2.0, "seed": 22},
        {"ming_latent_history": history, "ming_cfg": 3.0, "seed": 33},
    ]

    results = LingV3TTSLLMModel._vectorized_decode_results(
        model,
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        request_infos,
        [1, 1, 1],
    )

    assert set(results) == {0, 1}
    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape == (2, 4)
    assert calls[0]["latent_history"].shape == (2, 2, 3)
    assert calls[0]["initial_noise"].shape == (2, 1, 3)
    assert calls[0]["solver_noise"].shape == (10, 2, 1, 3)
    assert calls[0]["cfg_scale"] == 2.0
    assert not torch.equal(calls[0]["initial_noise"][0], calls[0]["initial_noise"][1])


def test_sft_config_preserves_gate_and_sampling_defaults():
    legacy = _config()
    backbone = legacy.llm_config.to_dict()
    backbone.update(kda_safe_gate=True, kda_lower_bound=-5.0)
    config = LingV3TTSConfig(
        llm_config=backbone,
        audio_config=legacy.audio_config,
        tts_config={
            **legacy.tts_config,
            "task_contract": "lingv3_sft_multitask_v1",
            "sampling_temperature": 1.5,
            "minimum_decode_steps": 13,
            "maximum_decode_steps": 250,
        },
    )
    runtime = LingV3TTSRuntimeConfig.from_hf_config(config)
    runtime.validate()
    assert config.llm_config.kda_safe_gate is True
    assert config.llm_config.kda_lower_bound == -5.0
    assert runtime.temperature == 1.5
    assert runtime.stop_head_min_steps == 13
    assert runtime.max_decode_steps == 250


def test_unknown_task_contract_is_rejected():
    with pytest.raises(ValueError, match="Unsupported LingV3 task contract"):
        LingV3TTSConfig(tts_config={"task_contract": "unknown"})


@pytest.mark.parametrize(
    "tts_config,ref_text,max_tokens,error",
    [
        ({}, "reference", 5, None),
        ({"conditioning_mode": "speaker_only"}, "reference", None, None),
        ({"conditioning_mode": "speaker_only"}, None, None, None),
        ({"minimum_decode_steps": 13}, None, 12, "at least 13"),
        ({"minimum_decode_steps": 13}, None, 13, None),
    ],
)
def test_adapter_validates_checkpoint_conditioning_and_minimum(tts_config, ref_text, max_tokens, error):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.lingv3_tts import LingV3TTSAdapter

    server = SimpleNamespace(
        engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(tts_config=tts_config))),
        _apply_uploaded_speaker=lambda request: None,
        _validate_ref_audio_format=lambda source: None,
    )
    adapter = LingV3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(
        input="Hello", ref_audio="data:audio/wav;base64,AA==", ref_text=ref_text, max_new_tokens=max_tokens
    )
    result = adapter.validate(request)
    if error is None:
        assert result is None
    else:
        assert error in result
