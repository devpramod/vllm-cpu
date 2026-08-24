# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np

from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer


class _FakeInputBatch:
    def __init__(self):
        self.req_ids = ["req-0", "req-1", "req-2"]
        self.req_id_to_index = {
            req_id: index for index, req_id in enumerate(self.req_ids)
        }
        self.num_prompt_tokens = np.array([2, 2, 2], dtype=np.int32)
        self.num_tokens_no_spec = np.array([3, 3, 8], dtype=np.int32)
        self.token_ids_cpu = np.array(
            [
                [1, 2, 3, 0, 0, 0, 0, 0],
                [4, 5, 6, 0, 0, 0, 0, 0],
                [7, 8, 9, 10, 11, 12, 13, 14],
            ],
            dtype=np.int32,
        )


class _FakeSuffixCache:
    def __init__(self):
        self.active_requests = {"stale"}
        self.cached_requests = {"req-0"}
        self.calls = []

    def evict_cached_response(self, req_id):
        self.calls.append(("evict", req_id))
        self.cached_requests.remove(req_id)

    def start_request(self, req_id, prompt_token_ids):
        self.calls.append(("start", req_id, prompt_token_ids.tolist()))
        self.active_requests.add(req_id)

    def add_active_response(self, req_id, sampled_token_ids):
        self.calls.append(("add", req_id, sampled_token_ids))

    def speculate(
        self,
        req_id,
        pattern,
        max_spec_tokens,
        max_spec_factor,
        min_token_prob,
    ):
        self.calls.append(
            (
                "speculate",
                req_id,
                pattern.tolist(),
                max_spec_tokens,
                max_spec_factor,
                min_token_prob,
            )
        )
        return SimpleNamespace(token_ids=[20, 21])

    def stop_request(self, req_id):
        self.calls.append(("stop", req_id))
        self.active_requests.remove(req_id)


def test_suffix_proposer_uses_shared_batch_lifecycle():
    proposer = SuffixDecodingProposer.__new__(SuffixDecodingProposer)
    proposer.num_speculative_tokens = 4
    proposer.max_model_len = 8
    proposer.max_tree_depth = 2
    proposer.max_spec_factor = 2.0
    proposer.min_token_prob = 0.1
    proposer.suffix_cache = _FakeSuffixCache()

    drafts = proposer.propose(4, _FakeInputBatch(), [[3], [], [14]])

    assert drafts == [[20, 21], [], []]
    assert proposer.suffix_cache.calls == [
        ("evict", "req-0"),
        ("start", "req-0", [1, 2]),
        ("add", "req-0", [3]),
        ("speculate", "req-0", [2, 3], 4, 2.0, 0.1),
        ("stop", "stale"),
    ]
