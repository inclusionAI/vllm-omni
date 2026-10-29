# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_omni.model_executor.models.lingv3_tts.lingv3_tts import LingV3TTSForConditionalGeneration
from vllm_omni.model_executor.models.lingv3_tts.lingv3_tts_llm import LingV3TTSLLMModel
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class FakeGraph:
    def __init__(self, runnable, config, runtime_mode):
        self.runnable = runnable
        self.runtime_mode = runtime_mode
        self.concrete_cudagraph_entries = {}

    def unwrap(self):
        return self.runnable

    def __getattr__(self, name):
        return getattr(self.runnable, name)

    def __call__(self, *args):
        return self.runnable(*args)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        return inputs_embeds * self.weight + positions.unsqueeze(-1)


def make_models():
    llm = LingV3TTSLLMModel.__new__(LingV3TTSLLMModel)
    torch.nn.Module.__init__(llm)
    llm.model = Backbone()
    llm._backbone_cudagraph = None
    outer = LingV3TTSForConditionalGeneration.__new__(LingV3TTSForConditionalGeneration)
    torch.nn.Module.__init__(outer)
    outer.model_stage = "llm"
    outer.model = llm
    return outer, llm


def test_full_scope_preserves_registered_backbone_and_outer_model(monkeypatch):
    outer, llm = make_models()
    keys = list(outer.state_dict())
    parameter = next(outer.parameters())
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.vllm_config = object()
    runner.model = FakeGraph(outer, runner.vllm_config, CUDAGraphMode.FULL)
    monkeypatch.setattr(
        "vllm_omni.worker.gpu_model_runner.current_omni_platform.get_graph_wrapper_cls", lambda: FakeGraph
    )
    runner._configure_model_cudagraph_scope()
    assert runner.model is outer
    assert llm._backbone_cudagraph.runnable is llm.model
    assert llm._backbone_cudagraph.runtime_mode == CUDAGraphMode.FULL
    assert list(outer.state_dict()) == keys
    assert next(outer.parameters()) is parameter
    for position in [1, 7]:
        out = llm._forward_backbone(torch.tensor([0]), torch.tensor([position]), None, torch.ones(1, 1))
        torch.testing.assert_close(out, torch.tensor([[float(position + 1)]]))


def test_piecewise_keeps_original_path():
    outer, llm = make_models()
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    runner.model = outer
    runner._configure_model_cudagraph_scope()
    assert runner.model is outer and llm._backbone_cudagraph is None
    torch.testing.assert_close(
        llm._forward_backbone(torch.tensor([0]), torch.tensor([3]), None, torch.ones(1, 1)), torch.tensor([[4.0]])
    )


def test_refuses_scope_change_after_capture(monkeypatch):
    outer, llm = make_models()
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.model = FakeGraph(outer, None, CUDAGraphMode.FULL)
    runner.model.concrete_cudagraph_entries["already-captured"] = object()
    monkeypatch.setattr(
        "vllm_omni.worker.gpu_model_runner.current_omni_platform.get_graph_wrapper_cls", lambda: FakeGraph
    )
    with pytest.raises(RuntimeError, match="before graph capture"):
        runner._configure_model_cudagraph_scope()
    assert llm._backbone_cudagraph is None


def test_other_models_keep_outer_full_wrapper():
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.model = FakeGraph(torch.nn.Linear(1, 1), None, CUDAGraphMode.FULL)
    before = runner.model
    runner._configure_model_cudagraph_scope()
    assert runner.model is before
