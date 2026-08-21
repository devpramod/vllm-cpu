# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import tokenize_template_drafts
from vllm.tokenizers.registry import get_tokenizer
from vllm.utils.cache import LRUCache
from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)

_TemplateTokenIds = tuple[tuple[int, ...], ...]


class _TemplateTrieNode:
    __slots__ = ("children", "template")

    def __init__(self):
        self.children: dict[int, _TemplateTrieNode] = {}
        self.template: np.ndarray | None = None


def _build_template_trie(templates: list[np.ndarray]) -> _TemplateTrieNode:
    root = _TemplateTrieNode()
    for template in templates:
        node = root
        for token_id in template:
            if node.template is None:
                node.template = template
            token = int(token_id)
            child = node.children.get(token)
            if child is None:
                child = _TemplateTrieNode()
                node.children[token] = child
            node = child
    return root


class _TemplateSet:
    __slots__ = ("max_template_len", "templates", "trie")

    def __init__(self, template_token_ids: _TemplateTokenIds):
        self.templates = [
            np.array(token_ids, dtype=np.int32) for token_ids in template_token_ids
        ]
        self.trie = _build_template_trie(self.templates)
        self.max_template_len = max(len(template) for template in self.templates)


class TemplateProposer:
    """
    Speculative decoding proposer for template-shaped responses.

    Intended for workloads whose responses are drawn from a small, known set
    of near-deterministic strings — e.g. guard/judge/classifier models that
    emit a fixed verdict format ("<think>\\n</think>\\n<score> yes </score>")
    differing only in a few decision tokens. For such workloads the optimal
    draft is not a model at all: propose the expected response directly and
    let rejection sampling verify it. A matching response is then produced in
    O(response_len / num_speculative_tokens) target forward passes, at zero
    drafting cost.

    Service-default templates are configured as plain strings
    (`template_drafts`) and tokenized with the target model's tokenizer at
    startup. Requests can inherit, disable, or override them. A draft is
    proposed if and only if the tokens generated so far are an exact prefix
    of one of the request's active templates; the proposal is that template's
    remainder, capped at `num_speculative_tokens`. Consequently:

    * Requests whose responses deviate from every template are never
      speculated again (exact-prefix matching fails from the divergence point
      onward), so mixed traffic pays no ongoing verification overhead.
    * When several templates share a prefix, the earliest one in
      `template_drafts` is proposed. If its decision token is rejected, the
      bonus token emitted by rejection sampling reveals the actual branch and
      the matching template takes over on the next step.

    Like all speculative decoding in vLLM, this is lossless: proposed tokens
    are only accepted when the target model's own sampling agrees with them.
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        assert config.template_drafts, "template_drafts must be set"
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        tokenizer = get_tokenizer(
            vllm_config.model_config.tokenizer,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )
        template_token_ids = tokenize_template_drafts(
            config.template_drafts, tokenizer, config.template_append_eos
        )
        self.default_template_token_ids = template_token_ids
        self.default_template_set = _TemplateSet(template_token_ids)
        self.template_set_cache: LRUCache[_TemplateTokenIds, _TemplateSet] = LRUCache(
            vllm_config.scheduler_config.max_num_seqs
        )

        self.templates = self.default_template_set.templates
        self.template_trie = self.default_template_set.trie
        self.max_template_len = self.default_template_set.max_template_len
        logger.info(
            "TemplateProposer initialized with %d template(s) of token "
            "lengths %s (append_eos=%s).",
            len(self.templates),
            [len(template) for template in self.templates],
            config.template_append_eos,
        )

    def _get_template_set(
        self, template_token_ids: _TemplateTokenIds | None
    ) -> _TemplateSet | None:
        if template_token_ids is None or (
            template_token_ids == self.default_template_token_ids
        ):
            return self.default_template_set
        if not template_token_ids:
            return None

        template_set = self.template_set_cache.get(template_token_ids)
        if template_set is None:
            template_set = _TemplateSet(template_token_ids)
            self.template_set_cache[template_token_ids] = template_set
        return template_set

    def propose(
        self,
        num_speculative_tokens: int,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
        request_template_token_ids: list[_TemplateTokenIds | None] | None = None,
    ) -> list[list[int]]:
        """
        Propose the remainder of the matching template for each request whose
        generated tokens so far are an exact prefix of one of the templates.
        Entries may have different lengths; requests with no matching
        template get an empty proposal.
        """
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            template_token_ids = (
                request_template_token_ids[i]
                if request_template_token_ids is not None
                else None
            )
            template_set = self._get_template_set(template_token_ids)
            if template_set is None:
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            index = input_batch.req_id_to_index[req_id]
            num_prompt_tokens = input_batch.num_prompt_tokens[index]
            num_output_tokens = num_tokens - num_prompt_tokens
            if num_output_tokens > template_set.max_template_len:
                # The response has outgrown every template.
                draft_token_ids.append([])
                continue

            response = input_batch.token_ids_cpu[i, num_prompt_tokens:num_tokens]
            max_tokens = min(
                num_speculative_tokens, self.max_model_len - num_tokens - 1
            )
            draft = _propose_template_remainder(response, template_set.trie, max_tokens)
            draft_token_ids.append(draft)
        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


def _propose_template_remainder(
    response: np.ndarray,
    template_trie: _TemplateTrieNode,
    max_tokens: int,
) -> list[int]:
    """
    Return the remainder of the first continuing template at `response`.

    The lookup takes O(len(response)) time. The result is capped at
    `max_tokens`; an empty list is returned when no template continues from
    the response or when `max_tokens` is not positive.
    """
    if max_tokens <= 0:
        return []
    node = template_trie
    for token_id in response:
        child = node.children.get(int(token_id))
        if child is None:
            return []
        node = child

    template = node.template
    if template is None:
        return []
    num_response_tokens = response.shape[0]
    remainder = template[num_response_tokens : num_response_tokens + max_tokens]
    return remainder.tolist()
