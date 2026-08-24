# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch

_RequestMetadataT = TypeVar("_RequestMetadataT")


class SpecDecodeNonModelProposer(ABC, Generic[_RequestMetadataT]):
    """Shared batch lifecycle for proposers that do not load a draft model."""

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

    def propose(
        self,
        num_speculative_tokens: int,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
        request_metadata: Sequence[_RequestMetadataT] | None = None,
    ) -> list[list[int]]:
        self._validate_num_speculative_tokens(num_speculative_tokens)
        if request_metadata is not None:
            assert len(request_metadata) == len(sampled_token_ids)

        draft_token_ids: list[list[int]] = []
        for request_index, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            num_tokens = input_batch.num_tokens_no_spec[request_index]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            metadata = (
                request_metadata[request_index]
                if request_metadata is not None
                else None
            )
            max_proposal_tokens = min(
                num_speculative_tokens, self.max_model_len - num_tokens - 1
            )
            draft_token_ids.append(
                self._propose_tokens(
                    input_batch,
                    request_index,
                    sampled_ids,
                    num_tokens,
                    max_proposal_tokens,
                    metadata,
                )
            )

        self._finish_batch(input_batch)
        return draft_token_ids

    def _validate_num_speculative_tokens(self, num_speculative_tokens: int) -> None:
        pass

    @abstractmethod
    def _propose_tokens(
        self,
        input_batch: InputBatch,
        request_index: int,
        sampled_token_ids: list[int],
        num_tokens: int,
        max_proposal_tokens: int,
        request_metadata: _RequestMetadataT | None,
    ) -> list[int]:
        pass

    def _finish_batch(self, input_batch: InputBatch) -> None:
        pass

    def load_model(self, *args, **kwargs):
        pass
