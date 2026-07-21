# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.tokenizers.registry import get_tokenizer
from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


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

    Templates are configured as plain strings (`template_drafts`) and
    tokenized with the target model's tokenizer at startup. A draft is
    proposed for a request if and only if the tokens generated so far are an
    exact prefix of one of the templates; the proposal is that template's
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
        eos_token_id = tokenizer.eos_token_id
        templates: list[np.ndarray] = []
        for template in config.template_drafts:
            token_ids = tokenizer.encode(template, add_special_tokens=False)
            if not token_ids:
                raise ValueError(
                    f"Template {template!r} tokenized to an empty sequence."
                )
            if config.template_append_eos and eos_token_id is not None:
                token_ids = token_ids + [eos_token_id]
            template_array = np.array(token_ids, dtype=np.int32)
            if any(np.array_equal(template_array, seen) for seen in templates):
                raise ValueError(
                    f"Template {template!r} duplicates an earlier template "
                    "after tokenization."
                )
            templates.append(template_array)
        self.templates = templates
        self.max_template_len = max(len(t) for t in templates)
        logger.info(
            "TemplateProposer initialized with %d template(s) of token "
            "lengths %s (append_eos=%s).",
            len(templates),
            [len(t) for t in templates],
            config.template_append_eos,
        )

    def propose(
        self,
        num_speculative_tokens: int,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
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

            req_id = input_batch.req_ids[i]
            index = input_batch.req_id_to_index[req_id]
            num_prompt_tokens = input_batch.num_prompt_tokens[index]
            num_output_tokens = num_tokens - num_prompt_tokens
            if num_output_tokens > self.max_template_len:
                # The response has outgrown every template.
                draft_token_ids.append([])
                continue

            response = input_batch.token_ids_cpu[i, num_prompt_tokens:num_tokens]
            max_tokens = min(
                num_speculative_tokens, self.max_model_len - num_tokens - 1
            )
            draft = _propose_template_remainder(response, self.templates, max_tokens)
            draft_token_ids.append(draft)
        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


def _propose_template_remainder(
    response: np.ndarray,
    templates: list[np.ndarray],
    max_tokens: int,
) -> list[int]:
    """
    Return the remainder of the first template whose prefix exactly equals
    `response`, capped at `max_tokens` tokens. Returns an empty list when no
    template matches (including when the response already equals a complete
    template) or when `max_tokens` is not positive.
    """
    if max_tokens <= 0:
        return []
    num_response_tokens = response.shape[0]
    for template in templates:
        if num_response_tokens < template.shape[0] and np.array_equal(
            template[:num_response_tokens], response
        ):
            remainder = template[num_response_tokens : num_response_tokens + max_tokens]
            return remainder.tolist()
    return []
