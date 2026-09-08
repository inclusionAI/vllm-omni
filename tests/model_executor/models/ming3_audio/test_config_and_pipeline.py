# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, PreTrainedTokenizerFast
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    is_hybrid,
    supports_multimodal,
    supports_pp,
    supports_transcription,
)
from vllm.multimodal.processing import InputProcessingContext
from vllm.tokenizers import cached_tokenizer_from_config

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config import OmniModelConfig
from vllm_omni.config.endpoint_policy import OmniServingCapability
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import StageExecutionType, load_deploy_config
from vllm_omni.model_executor.models.ming3_audio import (
    ming3_audio as ming3_audio_module,
)
from vllm_omni.model_executor.models.ming3_audio.ming3_audio import (
    BailingMoeV3AudioForConditionalGeneration,
    Ming3AudioStreamingPostProcessor,
)
from vllm_omni.model_executor.models.registry import OmniModelRegistry
from vllm_omni.transformers_utils.configs.ming3_audio import (
    BailingMoeV3AudioConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a self-contained one-layer checkpoint layout for CPU CI."""
    model_path = tmp_path_factory.mktemp("ming3_audio")
    config = BailingMoeV3AudioConfig(
        architectures=["BailingMoeV3AudioForConditionalGeneration"],
        text_config={
            "model_type": "",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 128,
            "head_dim": 8,
            "qk_head_dim": 8,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
            "kv_lora_rank": 4,
        },
        audio_config={
            # Whisper's feature extractor supports 80- and 128-bin mels.
            "n_mels": 80,
            "n_audio_ctx": 128,
            "n_audio_state": 8,
            "n_audio_head": 2,
            "n_audio_layer": 1,
            "downsampler_kernel_size_list": [3, 3, 3],
            "downsampler_stride_list": [2, 2, 2],
            "downsampler_padding_list": [1, 1, 1],
            "whisper_chunk_size": 32,
        },
        audio_token_id=3,
    )
    config.save_pretrained(model_path)
    # Match the released checkpoint's unusual empty nested model_type. The
    # native adapter must normalize it without loading remote configuration.
    config_path = model_path / "config.json"
    serialized_config = json.loads(config_path.read_text())
    serialized_config["text_config"]["model_type"] = ""
    config_path.write_text(json.dumps(serialized_config))

    backend = Tokenizer(
        WordLevel(
            vocab={
                "[UNK]": 0,
                "<|audio_start|>": 1,
                "<|audio_end|>": 2,
                "<|audio_pad|>": 3,
                "test": 4,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=[
            "<|audio_start|>",
            "<|audio_end|>",
            "<|audio_pad|>",
        ],
    )
    tokenizer.save_pretrained(model_path)
    return model_path


@pytest.fixture(scope="module")
def tiny_model_config(tiny_model_path: Path) -> OmniModelConfig:
    return OmniModelConfig(
        model=str(tiny_model_path),
        tokenizer=str(tiny_model_path),
        trust_remote_code=False,
        max_model_len=128,
        limit_mm_per_prompt={"audio": 1},
        model_arch="BailingMoeV3AudioForConditionalGeneration",
        model_stage="asr",
        hf_config_name="text_config",
    )


@pytest.fixture(scope="module")
def tiny_processor(tiny_model_config: OmniModelConfig):
    ctx = InputProcessingContext(
        tiny_model_config,
        tokenizer=cached_tokenizer_from_config(tiny_model_config),
    )
    return BailingMoeV3AudioForConditionalGeneration._processor_factory.build_processor(ctx, cache=None)


@pytest.fixture
def tiny_model(
    tiny_model_config: OmniModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> BailingMoeV3AudioForConditionalGeneration:
    class DummyLanguageModel(nn.Module):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            super().__init__()
            self.vllm_config = vllm_config
            self.prefix = prefix
            text_config = vllm_config.model_config.hf_config
            self.embedding = nn.Embedding(text_config.vocab_size, text_config.hidden_size)

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embedding(input_ids)

        def make_empty_intermediate_tensors(self, *args: object, **kwargs: object):
            return None

    monkeypatch.setattr(
        ming3_audio_module,
        "BailingMoeV3ForCausalLM",
        DummyLanguageModel,
    )
    return BailingMoeV3AudioForConditionalGeneration(vllm_config=VllmConfig(model_config=tiny_model_config))


def test_native_config_loads_without_remote_code(tiny_model_path: Path):
    config = AutoConfig.from_pretrained(tiny_model_path, trust_remote_code=False)

    assert isinstance(config, BailingMoeV3AudioConfig)
    assert config.architectures == ["BailingMoeV3AudioForConditionalGeneration"]
    assert config.model_type == "bailing_moe_v3_audio"
    assert config.text_config.model_type == "bailing_hybrid"
    assert config.text_config.hidden_size == 16
    assert config.text_config.num_hidden_layers == 1
    assert config.audio_config.n_audio_layer == 1
    assert config.audio_config.downsampler_stride_list == [2, 2, 2]
    assert config.audio_token_id == 3
    assert config.norm_query_embeds is True
    assert config.get_text_config() is config.text_config


def test_native_audio_preprocessing_without_remote_code(
    tiny_model_config: OmniModelConfig,
    tiny_processor,
):
    assert tiny_model_config.use_mla is True
    assert tiny_model_config.get_head_size() == 8
    placeholder = BailingMoeV3AudioForConditionalGeneration.get_placeholder_str("audio", 0)
    items = tiny_processor.info.parse_mm_data({"audio": (np.zeros(16000, dtype=np.float32), 16000)})
    output = tiny_processor(
        placeholder + "test",
        mm_items=items,
        hf_processor_mm_kwargs={},
    )
    mm_data = output["mm_kwargs"].get_data()

    assert output["prompt_token_ids"].count(tiny_model_config.hf_config.audio_token_id) == 7
    assert mm_data["audio_feats"].shape == (1, 100, 80)
    assert mm_data["audio_feats_lengths"].tolist() == [[100]]


def test_partial_dummy_model_executes_real_audio_path(
    tiny_model: BailingMoeV3AudioForConditionalGeneration,
):
    assert len(tiny_model.audio_encoder.blocks) == 1
    assert tiny_model.language_model.prefix == "language_model"
    assert tiny_model.language_model.vllm_config.model_config.hf_config.num_hidden_layers == 1

    embeddings = tiny_model.embed_multimodal(
        audio_feats=torch.zeros(1, 63, 80),
        audio_feats_lengths=torch.tensor([[63]]),
    )

    assert len(embeddings) == 1
    assert embeddings[0].shape == (4, 16)
    assert torch.isfinite(embeddings[0]).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(embeddings[0], dim=-1),
        torch.ones(4),
    )


def test_partial_dummy_model_loads_mapped_audio_weights(
    tiny_model: BailingMoeV3AudioForConditionalGeneration,
):
    encoder_weight = torch.full_like(tiny_model.audio_encoder.conv1.weight, 0.25)
    projector_weight = torch.full_like(tiny_model.audio_projector.proj[0].weight, 0.5)

    loaded = tiny_model.load_weights(
        [
            ("model.audio.conv1.weight", encoder_weight),
            ("linear_proj_audio.0.weight", projector_weight),
            ("model.visual.blocks.0.weight", torch.ones(1)),
        ]
    )

    assert loaded == {
        "audio_encoder.conv1.weight",
        "audio_projector.proj.0.weight",
    }
    torch.testing.assert_close(tiny_model.audio_encoder.conv1.weight, encoder_weight)
    torch.testing.assert_close(tiny_model.audio_projector.proj[0].weight, projector_weight)


def test_ming3_model_is_registered():
    model_cls = OmniModelRegistry._try_load_model_cls("BailingMoeV3AudioForConditionalGeneration")
    assert model_cls is BailingMoeV3AudioForConditionalGeneration


def test_ming3_advertises_required_vllm_capabilities():
    model = BailingMoeV3AudioForConditionalGeneration
    assert supports_multimodal(model)
    assert supports_pp(model)
    assert is_hybrid(model)
    assert supports_transcription(model)


def test_checkpoint_prefix_mapping_reuses_bailing_loader():
    mapper = BailingMoeV3AudioForConditionalGeneration.hf_to_vllm_mapper
    names = mapper.apply_list(
        [
            "model.audio.conv1.weight",
            "linear_proj_audio.0.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
            "model.visual.blocks.0.weight",
            "linear_proj.0.weight",
        ]
    )
    assert names == [
        "audio_encoder.conv1.weight",
        "audio_projector.proj.0.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.lm_head.weight",
    ]


def test_asr_service_does_not_split_model_context():
    config = BailingMoeV3AudioForConditionalGeneration.get_speech_to_text_config(None, "transcribe")
    assert config.sample_rate == 16000
    assert config.max_audio_clip_s is None
    assert config.min_energy_split_window_size is None
    assert config.allow_audio_chunking is False

    translation_init_config = BailingMoeV3AudioForConditionalGeneration.get_speech_to_text_config(None, "translate")
    assert translation_init_config.sample_rate == 16000


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("English\thello world", "hello world"),
        ("hello world", "hello world"),
    ],
)
def test_language_prefix_post_processing(text: str, expected: str):
    assert BailingMoeV3AudioForConditionalGeneration.post_process_output(text) == expected


def test_streaming_language_prefix_post_processing():
    processor = Ming3AudioStreamingPostProcessor()
    assert processor.process_delta("English", finished=False) == ""
    assert processor.process_delta("\thello", finished=False) == "hello"
    assert processor.process_delta(" world", finished=True) == " world"


def test_ming3_pipeline_is_single_stage_asr():
    pipeline = resolve_pipeline_config("bailing_moe_v3_audio")

    assert pipeline is not None
    assert pipeline.model_arch == "BailingMoeV3AudioForConditionalGeneration"
    assert pipeline.default_deploy_config_name == "ming3_audio_flash.yaml"
    assert len(pipeline.endpoint_restrictions) == 1
    assert pipeline.endpoint_restrictions[0].capability is OmniServingCapability.TRANSLATIONS
    assert len(pipeline.stages) == 1

    stage = pipeline.get_stage(0)
    assert stage.model_stage == "asr"
    assert stage.execution_type is StageExecutionType.LLM_AR
    assert stage.input_sources == ()
    assert stage.final_output is True
    assert stage.final_output_type == "text"
    assert stage.engine_output_type == "text"
    assert stage.owns_tokenizer is True
    assert stage.requires_multimodal_data is True
    assert stage.hf_config_name == "text_config"
    assert stage.sampling_constraints == {"detokenize": True}


def test_ming3_default_deploy_is_conservative_four_gpu_profile():
    deploy = load_deploy_config(get_deploy_config_path("ming3_audio_flash.yaml"))

    assert deploy.pipeline == "bailing_moe_v3_audio"
    assert deploy.dtype == "bfloat16"
    assert deploy.enable_chunked_prefill is True
    assert deploy.enable_prefix_caching is False
    assert len(deploy.stages) == 1

    stage = deploy.stages[0]
    assert stage.devices == "0,1,2,3"
    assert stage.tensor_parallel_size == 4
    assert stage.enable_expert_parallel is False
    assert stage.max_num_seqs == 1
    assert stage.max_num_batched_tokens == 2048
    assert stage.gpu_memory_utilization == 0.92
    assert stage.enforce_eager is True
