# Template Drafting

The following code configures vLLM to use speculative decoding where proposals come from a small, user-supplied set of response templates.

Template drafting targets workloads whose responses are drawn from a known set of near-deterministic strings: guard, judge, and classifier models that emit a fixed verdict format, routers that answer with one of a few labels, or any deployment where the model's output is a short structured decision. For such workloads the best draft is not a model at all — the expected response is proposed directly and verified by the target model, so a matching response completes in one or two target forward passes at zero drafting cost.

A draft is proposed for a request only while its generated tokens are an exact prefix of one of its active templates. The service-level templates are the default, but each request can inherit, disable, or override them. Requests whose responses deviate from every active template stop being speculated from the point of divergence, so mixed traffic pays no ongoing overhead. When several templates share a common prefix (e.g. verdicts differing only in a `yes`/`no` token), the earliest template in the list is proposed; if the decision token is rejected, the token emitted by rejection sampling reveals the actual branch and the matching template takes over on the next step.

Like all speculative decoding in vLLM, template drafting is lossless: proposed tokens are accepted only when the target model's own sampling agrees with them, so outputs are identical to running without speculation.

!!! tip "Choosing `num_speculative_tokens`"
    Proposals are capped at the matched template's remaining length, so set `num_speculative_tokens` to at least the token length of your longest template (plus one for the EOS token appended by default) to verify a full response in a single step.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="ibm-granite/granite-guardian-4.1-8b",
    speculative_config={
        "method": "template",
        "template_drafts": [
            "<think>\n</think>\n<score> yes </score>",
            "<think>\n</think>\n<score> no </score>",
        ],
        "num_speculative_tokens": 12,
    },
)
outputs = llm.generate(["..."], SamplingParams(temperature=0))
```

Configuration:

- `template_drafts`: service-level default response templates as plain strings, tokenized with the target model's tokenizer at startup. List the most likely template first; ties on shared prefixes go to the earliest entry.
- `template_append_eos` (default `true`): append the tokenizer's EOS token to each template, letting the final verification step accept the end of the response as well.

## Per-request templates

Set `SamplingParams.template_drafts` to control template drafting for an individual request:

- `None` (default): use the service-level templates.
- `[]`: disable template drafting for this request.
- A non-empty list: override the service-level templates for this request.

Request overrides are tokenized once when the request enters the engine and inherit the service-level `template_append_eos` setting.

```python
outputs = llm.generate(
    ["classify without thinking", "reason before classifying"],
    [
        SamplingParams(temperature=0),
        SamplingParams(temperature=0, template_drafts=[]),
    ],
)
```

The OpenAI-compatible Chat Completions, Completions, and Responses APIs expose the same field. Pass it through `extra_body` when using the OpenAI Python client:

```python
response = client.chat.completions.create(
    model="ibm-granite/granite-guardian-4.1-8b",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"template_drafts": []},
)
```

Template selection is independent of `reasoning_effort` and `chat_template_kwargs.enable_thinking`. Set both explicitly when switching between think and no-think modes so that the active templates match the output format requested from the model.

Template drafting composes with prefix caching and works on any hardware backend, since the proposer runs entirely on the CPU without a draft model.
