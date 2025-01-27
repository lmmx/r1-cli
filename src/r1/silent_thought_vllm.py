from __future__ import annotations

import json
import re
from typing import Literal, TypeVar

import argh
import torch
from pydantic import BaseModel, create_model
from vllm import LLM, SamplingParams

bot, eot = "<think>", "</think>"

GREEN = "\033[32m"
RESET = "\033[0m"


class TriggerBasedLogitsProcessor:
    """Logits processor that triggers JSON generation after </think> token"""

    def __init__(self, tokenizer, base_processor, guide: type[BaseModel]):
        self.tokenizer = tokenizer
        self.base_processor = base_processor
        self.guide = guide
        self.bot_id, self.eot_id = tokenizer.convert_tokens_to_ids([bot, eot])
        self.triggered = False
        self.triggered_at = -1
        self.in_cot = False
        self.history = []
        self.cot = ""

    def __call__(
        self, prompt: tuple[int], generated_tokens: tuple[int], logits: torch.Tensor
    ) -> torch.Tensor:
        if not self.in_cot and not self.cot:
            # We may be in the CoT if the BOT was in the prompt (but not EOT)
            self.in_cot = self.bot_id in prompt and self.eot_id not in prompt
            if self.in_cot:
                self.cot = self.tokenizer.decode(
                    prompt[prompt.index(self.bot_id) + 1 :]
                ).lstrip()
        if len(generated_tokens) > 0:
            last_id = generated_tokens[-1]
            if self.in_cot:
                is_eot = last_id == self.eot_id
                if is_eot:
                    self.triggered_at = len(generated_tokens)
                    self.triggered = True
                    self.in_cot = False
                    self.cot += self.tokenizer.decode(self.history).strip()
            self.history.append(last_id)

        # Only apply base processor if triggered
        if self.triggered:
            return self.base_processor(generated_tokens, logits)
        return logits


class StructuredAnswer(BaseModel):
    answer: str | int | float


def load_model(model_size: str):
    model_name = f"casperhansen/deepseek-r1-distill-qwen-{model_size}-awq"
    return LLM(model_name, enable_prefix_caching=True)


def think(
    messages: list[str],
    guide: type[BaseModel],
    model_size: Literal["1.5b", "7b", "14b", "32b"] = "7b",
    cot_prefill: str = "<think>\nOkay, so ",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    from outlines.models.vllm import adapt_tokenizer
    from outlines.processors import JSONLogitsProcessor
    from transformers import AutoTokenizer

    llm = load_model(model_size)
    tokenizer = llm.get_tokenizer()

    # Build the prompt with CoT markers
    msg_list = [{"role": "user", "content": msg} for msg in messages]
    prompt = (
        tokenizer.apply_chat_template(
            msg_list, tokenize=False, add_generation_prompt=True
        )
        + f"{cot_prefill}"
    )

    # Configure processors
    json_schema = json.dumps(guide.model_json_schema())
    model_name = llm.llm_engine.model_config.model
    outlines_tokenizer = adapt_tokenizer(AutoTokenizer.from_pretrained(model_name))
    guided_processor = JSONLogitsProcessor(
        schema=json_schema, tokenizer=outlines_tokenizer, whitespace_pattern=r" ?"
    )
    conditional_guide_processor = TriggerBasedLogitsProcessor(
        tokenizer=outlines_tokenizer, base_processor=guided_processor, guide=guide
    )
    # logits_processor = TriggerBasedLogitsProcessor(tokenizer, guide)
    # logits_processor = SimpleLogitsProcessor(tokenizer, guide)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        logits_processors=[conditional_guide_processor],
    )
    # Generate output
    logits_processor = sampling_params.logits_processors[
        0
    ]  # or conditional_guide_processor
    output = llm.generate(prompt, sampling_params, use_tqdm=False)
    generated_text = output[0].outputs[0].text
    cot = logits_processor.cot
    # JSON structured response here
    post_cot = tokenizer.decode(
        logits_processor.history[logits_processor.triggered_at :],
        skip_special_tokens=True,
    )
    structured = (
        '{"reasoning": "' + json.dumps(cot)[1:-1] + '", ' + post_cot.removeprefix("{")
    )
    return structured


def silencio(
    message: str,
    *,
    answer_type: Literal["int", "str", "float"],
    model_size: Literal["1.5B", "7B", "14B", "32B"] = "7B",
    cot_prefill: str = "<think>\nOkay, so ",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    return_result: bool = False,
):
    message = (
        "Consider the following question which I want you to answer:"
        f"\n\n'''{message}'''\n\n"
        f"Provide your answer as a {answer_type} value in JSON format after careful thinking."
    )

    answer_model = create_answer_model(answer_type)
    result = think(
        messages=[message],
        guide=answer_model,
        model_size=model_size,
        cot_prefill=cot_prefill,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    print(GREEN + result + RESET)
    if return_result:
        return result


def create_answer_model(answer_type: str) -> type[BaseModel]:
    type_map = {"float": float, "int": int, "str": str}
    return create_model(
        "AnswerModel", answer=(type_map[answer_type], ...), __base__=StructuredAnswer
    )


def main():
    argh.dispatch_command(silencio, old_name_mapping_policy=False)


if __name__ == "__main__":
    main()
