# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode.template_proposer import (
    TemplateProposer,
    _build_template_trie,
    _propose_template_remainder,
)

# Token-id shorthand for a guard-model-style verdict template pair that
# shares a prefix and diverges at one decision token:
#   yes: [10, 11, 12, 100, 20, 21, 0]
#   no:  [10, 11, 12, 200, 20, 21, 0]
YES = np.array([10, 11, 12, 100, 20, 21, 0], dtype=np.int32)
NO = np.array([10, 11, 12, 200, 20, 21, 0], dtype=np.int32)
TEMPLATES = [YES, NO]
TEMPLATE_TRIE = _build_template_trie(TEMPLATES)


def _response(*token_ids: int) -> np.ndarray:
    return np.array(token_ids, dtype=np.int32)


def test_response_start_proposes_first_template():
    # Empty response matches every template; the first one wins the tie.
    assert _propose_template_remainder(_response(), TEMPLATE_TRIE, 16) == YES.tolist()


def test_mid_template_proposes_remainder():
    assert _propose_template_remainder(_response(10, 11), TEMPLATE_TRIE, 16) == [
        12,
        100,
        20,
        21,
        0,
    ]


def test_branch_correction_selects_matching_template():
    # After the decision token diverges to the second branch (e.g. via the
    # bonus token on rejection), the second template must take over.
    assert _propose_template_remainder(
        _response(10, 11, 12, 200), TEMPLATE_TRIE, 16
    ) == [20, 21, 0]


def test_non_template_response_proposes_nothing():
    assert _propose_template_remainder(_response(10, 99), TEMPLATE_TRIE, 16) == []


def test_complete_template_proposes_nothing():
    assert (
        _propose_template_remainder(_response(*YES.tolist()), TEMPLATE_TRIE, 16) == []
    )


def test_max_tokens_caps_proposal():
    assert _propose_template_remainder(_response(10), TEMPLATE_TRIE, 2) == [11, 12]
    assert _propose_template_remainder(_response(10), TEMPLATE_TRIE, 0) == []


def test_multiple_branch_levels_preserve_template_order():
    templates = [
        _response(1, 2, 3, 4),
        _response(1, 2, 30, 31),
        _response(1, 20, 21, 22),
    ]
    trie = _build_template_trie(templates)

    assert _propose_template_remainder(_response(1), trie, 16) == [2, 3, 4]
    assert _propose_template_remainder(_response(1, 2), trie, 16) == [3, 4]
    assert _propose_template_remainder(_response(1, 2, 30), trie, 16) == [31]
    assert _propose_template_remainder(_response(1, 20), trie, 16) == [21, 22]


def test_completed_template_does_not_block_longer_template():
    templates = [_response(1, 2), _response(1, 2, 3)]
    trie = _build_template_trie(templates)

    assert _propose_template_remainder(_response(1), trie, 16) == [2]
    assert _propose_template_remainder(_response(1, 2), trie, 16) == [3]
    assert _propose_template_remainder(_response(1, 2, 3), trie, 16) == []


def test_trie_matches_linear_reference_for_large_template_family():
    templates = [
        _response(10, branch // 8, branch % 8, 20, branch) for branch in range(64)
    ]
    trie = _build_template_trie(templates)

    def linear_reference(response: np.ndarray, max_tokens: int) -> list[int]:
        for template in templates:
            if response.shape[0] < template.shape[0] and np.array_equal(
                template[: response.shape[0]], response
            ):
                return template[
                    response.shape[0] : response.shape[0] + max_tokens
                ].tolist()
        return []

    responses = [
        _response(),
        _response(10),
        _response(10, 7),
        _response(10, 7, 7),
        _response(10, 7, 7, 20, 63),
        _response(10, 99),
    ]
    for response in responses:
        assert _propose_template_remainder(response, trie, 3) == linear_reference(
            response, 3
        )


class _FakeTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens
        return {"yes": [10, 11, 12, 100, 20, 21], "no": [10, 11, 12, 200, 20, 21]}[text]


class _FakeInputBatch:
    def __init__(self, prompt_lens: list[int], token_rows: list[list[int]]):
        self.req_ids = [f"req-{i}" for i in range(len(token_rows))]
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(self.req_ids)}
        self.num_prompt_tokens = np.array(prompt_lens, dtype=np.int32)
        self.num_tokens_no_spec = np.array(
            [len(row) for row in token_rows], dtype=np.int32
        )
        max_len = max(len(row) for row in token_rows) + 8
        self.token_ids_cpu = np.zeros((len(token_rows), max_len), dtype=np.int32)
        for i, row in enumerate(token_rows):
            self.token_ids_cpu[i, : len(row)] = row


@pytest.fixture
def proposer(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.spec_decode.template_proposer.get_tokenizer",
        lambda *args, **kwargs: _FakeTokenizer(),
    )
    return TemplateProposer(
        VllmConfig(
            model_config=ModelConfig(model="facebook/opt-125m"),
            speculative_config=SpeculativeConfig(
                method="template",
                template_drafts=["yes", "no"],
                num_speculative_tokens=8,
            ),
        )
    )


def test_proposer_tokenizes_and_appends_eos(proposer):
    assert [t.tolist() for t in proposer.templates] == [YES.tolist(), NO.tolist()]


def test_proposer_batch(proposer):
    prompt = [1, 2, 3]
    batch = _FakeInputBatch(
        prompt_lens=[3, 3, 3, 3],
        token_rows=[
            prompt,  # response not started -> full first template
            prompt + [10, 11, 12, 200],  # "no" branch -> its remainder
            prompt + [10, 99],  # deviated -> no proposal
            prompt + YES.tolist(),  # complete -> no proposal
        ],
    )
    sampled = [[3], [200], [99], [0]]
    drafts = proposer.propose(8, batch, sampled)
    assert drafts == [
        YES.tolist()[:8],
        [20, 21, 0],
        [],
        [],
    ]


def test_proposer_uses_request_scoped_template_sets(proposer):
    prompt = [1, 2, 3]
    batch = _FakeInputBatch(
        prompt_lens=[3, 3, 3],
        token_rows=[
            prompt + [10, 11],
            prompt + [10, 11],
            prompt + [30, 31],
        ],
    )
    custom_templates = ((30, 31, 32, 33, 34, 35, 36, 37, 0),)

    drafts = proposer.propose(
        8,
        batch,
        [[11], [11], [31]],
        request_template_token_ids=[None, (), custom_templates],
    )

    assert drafts == [[12, 100, 20, 21, 0], [], [32, 33, 34, 35, 36, 37, 0]]
    assert proposer._get_template_set(custom_templates) is proposer._get_template_set(
        custom_templates
    )
    assert len(proposer.template_set_cache) == 1


def test_proposer_skips_partial_prefills_and_long_responses(proposer):
    prompt = [1, 2, 3]
    batch = _FakeInputBatch(
        prompt_lens=[3, 3],
        token_rows=[
            prompt + [10, 11],
            # Response longer than every template.
            prompt + YES.tolist() + [42, 43],
        ],
    )
    # Empty sampled ids (partial prefill) must not be speculated.
    drafts = proposer.propose(8, batch, [[], [43]])
    assert drafts == [[], []]


def test_duplicate_templates_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        SpeculativeConfig(
            method="template",
            template_drafts=["yes", "yes"],
            num_speculative_tokens=8,
        )


def test_templates_with_duplicate_tokenization_rejected(monkeypatch):
    class _DuplicateTokenizer:
        eos_token_id = None

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            assert not add_special_tokens
            return [10, 11]

    monkeypatch.setattr(
        "vllm.v1.spec_decode.template_proposer.get_tokenizer",
        lambda *args, **kwargs: _DuplicateTokenizer(),
    )
    config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m"),
        speculative_config=SpeculativeConfig(
            method="template",
            template_drafts=["yes", "yes with whitespace"],
            template_append_eos=False,
            num_speculative_tokens=8,
        ),
    )

    with pytest.raises(ValueError, match="duplicates an earlier template"):
        TemplateProposer(config)


def test_empty_templates_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        SpeculativeConfig(method="template", template_drafts=[])
