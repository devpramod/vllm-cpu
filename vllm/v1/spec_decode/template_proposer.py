# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import tokenize_template_drafts
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens,
    detokenize_incrementally,
)
from vllm.tokenizers.registry import get_tokenizer
from vllm.utils.cache import LRUCache
from vllm.v1.spec_decode.non_model_proposer import SpecDecodeNonModelProposer
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
    __slots__ = ("templates", "trie")

    def __init__(self, template_token_ids: _TemplateTokenIds):
        self.templates = [
            np.array(token_ids, dtype=np.int32) for token_ids in template_token_ids
        ]
        self.trie = _build_template_trie(self.templates)


class TemplateProposer(SpecDecodeNonModelProposer[_TemplateTokenIds | None]):
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
        proposed while the generated output is a prefix of one of the request's
        active templates. Exact token prefixes use a trie; if tokenization changes
        at an intermediate character boundary, a fallback proposes the remainder
        only after verifying that it reconstructs the same template text.
        Consequently:

        * Requests whose rendered responses deviate from every template receive
            no proposal. Ambiguous or unsupported token reconstructions also fail
            closed without affecting target-model output.
    * When several templates share a prefix, the earliest one in
      `template_drafts` is proposed. If its decision token is rejected, the
      bonus token emitted by rejection sampling reveals the actual branch and
      the matching template takes over on the next step.

    Like all speculative decoding in vLLM, this is lossless: proposed tokens
    are only accepted when the target model's own sampling agrees with them.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        assert config.template_drafts, "template_drafts must be set"

        self.tokenizer = get_tokenizer(
            vllm_config.model_config.tokenizer,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )
        template_token_ids = tokenize_template_drafts(
            config.template_drafts, self.tokenizer, config.template_append_eos
        )
        self.template_eos_token_id = (
            self.tokenizer.eos_token_id if config.template_append_eos else None
        )
        self.default_template_token_ids = template_token_ids
        self.default_template_set = _TemplateSet(template_token_ids)
        self.template_set_cache: LRUCache[_TemplateTokenIds, _TemplateSet] = LRUCache(
            vllm_config.scheduler_config.max_num_seqs
        )

        self.templates = self.default_template_set.templates
        self.template_trie = self.default_template_set.trie
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

    def _propose_tokens(
        self,
        input_batch: InputBatch,
        request_index: int,
        sampled_token_ids: list[int],
        num_tokens: int,
        max_proposal_tokens: int,
        request_metadata: _TemplateTokenIds | None,
    ) -> list[int]:
        template_set = self._get_template_set(request_metadata)
        if template_set is None:
            return []

        req_id = input_batch.req_ids[request_index]
        index = input_batch.req_id_to_index[req_id]
        num_prompt_tokens = input_batch.num_prompt_tokens[index]
        response = input_batch.token_ids_cpu[
            request_index, num_prompt_tokens:num_tokens
        ]
        draft = _propose_template_remainder(
            response, template_set.trie, max_proposal_tokens
        )
        if draft or max_proposal_tokens <= 0:
            return draft

        if not input_batch.is_token_ids[request_index, :num_prompt_tokens].all():
            return []
        prompt_token_ids = input_batch.token_ids_cpu[request_index, :num_prompt_tokens]
        return _propose_character_template_remainder(
            self.tokenizer,
            prompt_token_ids,
            response,
            template_set.templates,
            self.template_eos_token_id,
            max_proposal_tokens,
        )


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


def _detokenize_output(
    tokenizer: TokenizerLike,
    prompt_token_ids: np.ndarray,
    output_token_ids: np.ndarray | list[int],
) -> str | None:
    prompt_ids = [int(token_id) for token_id in prompt_token_ids]
    output_ids = [int(token_id) for token_id in output_token_ids]
    all_input_ids = prompt_ids.copy()
    try:
        decoded = tokenizer.decode(
            [*prompt_ids, *output_ids], skip_special_tokens=False
        )
        if "\ufffd" in decoded:
            return None

        tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
            tokenizer, prompt_ids, skip_special_tokens=False
        )
        output_text = ""
        for token_id in output_ids:
            all_input_ids.append(token_id)
            new_tokens, new_text, prefix_offset, read_offset = detokenize_incrementally(
                tokenizer,
                all_input_ids,
                tokens,
                prefix_offset,
                read_offset,
                skip_special_tokens=False,
            )
            tokens.extend(new_tokens)
            output_text += new_text
    except Exception:
        return None

    return output_text if "\ufffd" not in output_text else None


def _propose_character_template_remainder(
    tokenizer: TokenizerLike,
    prompt_token_ids: np.ndarray,
    response: np.ndarray,
    templates: list[np.ndarray],
    eos_token_id: int | None,
    max_tokens: int,
) -> list[int]:
    """Return a verified continuation for a character-equivalent prefix."""
    response_text = _detokenize_output(tokenizer, prompt_token_ids, response)
    if response_text is None or (response.size and not response_text):
        return []

    response_ids = [int(token_id) for token_id in response]
    for template in templates:
        has_appended_eos = (
            eos_token_id is not None
            and template.size > 0
            and int(template[-1]) == eos_token_id
        )
        content_token_ids = template[:-1] if has_appended_eos else template
        template_text = _detokenize_output(
            tokenizer, prompt_token_ids, content_token_ids
        )
        if template_text is None or not template_text.startswith(response_text):
            continue

        remaining_text = template_text[len(response_text) :]
        try:
            candidate_ids = (
                tokenizer.encode(remaining_text, add_special_tokens=False)
                if remaining_text
                else []
            )
        except Exception:
            continue

        reconstructed_text = _detokenize_output(
            tokenizer, prompt_token_ids, [*response_ids, *candidate_ids]
        )
        if reconstructed_text != template_text:
            continue

        if has_appended_eos:
            assert eos_token_id is not None
            candidate_ids.append(eos_token_id)
        return candidate_ids[:max_tokens]

    return []
