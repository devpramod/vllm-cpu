# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import VllmConfig
from vllm.v1.spec_decode.non_model_proposer import SpecDecodeNonModelProposer
from vllm.v1.worker.gpu_input_batch import InputBatch


class SuffixDecodingProposer(SpecDecodeNonModelProposer[None]):
    """
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference).
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob

        # Lazy import to avoid error when Suffix Decoding is not used.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )

    def _validate_num_speculative_tokens(self, num_speculative_tokens: int) -> None:
        assert num_speculative_tokens == self.num_speculative_tokens

    def _propose_tokens(
        self,
        input_batch: InputBatch,
        request_index: int,
        sampled_token_ids: list[int],
        num_tokens: int,
        max_proposal_tokens: int,
        request_metadata: None,
    ) -> list[int]:
        req_id = input_batch.req_ids[request_index]
        index = input_batch.req_id_to_index[req_id]
        if req_id not in self.suffix_cache.active_requests:
            if req_id in self.suffix_cache.cached_requests:
                # Reset the suffix cache for this request.
                self.suffix_cache.evict_cached_response(req_id)
            num_prompt_tokens = input_batch.num_prompt_tokens[index]
            prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
            # Start a new request, this will build the suffix tree for that prompt.
            self.suffix_cache.start_request(req_id, prompt_token_ids)

        # Append the newly sampled ids to the suffix cache for this request.
        self.suffix_cache.add_active_response(req_id, sampled_token_ids)

        # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
        # we extract the pattern from the end of the input.
        start = max(0, num_tokens - self.max_tree_depth)
        pattern = input_batch.token_ids_cpu[request_index, start:num_tokens]
        draft = self.suffix_cache.speculate(
            req_id,
            pattern,
            max_spec_tokens=max_proposal_tokens,
            max_spec_factor=self.max_spec_factor,
            min_token_prob=self.min_token_prob,
        )
        return draft.token_ids

    def _finish_batch(self, input_batch: InputBatch) -> None:
        # Stop requests that were not seen in the input batch.
        for req_id in (
            self.suffix_cache.active_requests - input_batch.req_id_to_index.keys()
        ):
            self.suffix_cache.stop_request(req_id)
