# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import msgspec
import pytest

from vllm import SamplingParams
from vllm.config import SpeculativeConfig


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(ValueError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


@pytest.mark.parametrize(
    ("template_drafts", "expected"),
    [(None, None), ([], []), (["yes", "no"], ["yes", "no"])],
)
def test_template_drafts_preserve_request_state(template_drafts, expected):
    params = SamplingParams(template_drafts=template_drafts)

    assert params.template_drafts == expected


@pytest.mark.parametrize(
    "template_drafts",
    ["yes", [""], ["yes", ""], ["yes", "yes"], ["yes", 1]],
)
def test_template_drafts_reject_invalid_values(template_drafts):
    with pytest.raises(ValueError, match="template_drafts"):
        SamplingParams(template_drafts=template_drafts)


def test_template_drafts_require_template_speculative_method():
    params = SamplingParams(template_drafts=[])

    with pytest.raises(ValueError, match="method='template'"):
        params.verify(MockModelConfig(), None, None, None)
    with pytest.raises(ValueError, match="method='template'"):
        params.verify(
            MockModelConfig(),
            SpeculativeConfig(method="ngram", num_speculative_tokens=4),
            None,
            None,
        )


def test_template_drafts_clone_and_msgspec_round_trip():
    params = SamplingParams(template_drafts=["yes", "no"])
    params._template_token_ids = ((10, 11), (20, 21))

    cloned = params.clone()
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(params), type=SamplingParams
    )

    assert cloned.template_drafts == params.template_drafts
    assert cloned._template_token_ids == params._template_token_ids
    assert decoded.template_drafts == params.template_drafts
    assert decoded._template_token_ids == params._template_token_ids


def test_template_drafts_tokenized_once_with_eos():
    class FakeTokenizer:
        eos_token_id = 0
        max_token_id = 100

        def encode(self, text, add_special_tokens=True):
            assert not add_special_tokens
            return {"yes": [10, 11], "no": [20, 21]}[text]

    params = SamplingParams(template_drafts=["yes", "no"])

    params.update_from_tokenizer(FakeTokenizer(), template_append_eos=True)

    assert params._template_token_ids == ((10, 11, 0), (20, 21, 0))


def test_template_drafts_reject_duplicate_tokenization():
    class FakeTokenizer:
        eos_token_id = None
        max_token_id = 100

        def encode(self, text, add_special_tokens=True):
            assert not add_special_tokens
            return [10, 11]

    params = SamplingParams(template_drafts=["yes", "yes with whitespace"])

    with pytest.raises(ValueError, match="duplicates an earlier template"):
        params.update_from_tokenizer(FakeTokenizer())
