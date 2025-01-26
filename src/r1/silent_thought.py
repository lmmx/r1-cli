from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Literal, TypeVar

import argh
from pydantic import BaseModel, create_model

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

bot, eot = "<think>", "</think>"


def hardcoded_schema(guide: type[PydanticModel], cot: str, strip: bool = True) -> str:
    """Take a model JSON schema, hardcode a constant string into it."""
    if strip:
        cot = cot.removeprefix(bot).removesuffix(eot).strip()
    hardcoded_cot = create_model("CoTModel", cot=(Literal[cot], ...))
    hc_schema = hardcoded_cot.model_json_schema()
    original_schema = guide.model_json_schema()
    original_schema["properties"]["reasoning"] = hc_schema["properties"]["cot"]
    return json.dumps(original_schema)


@lru_cache
def load_r1(model_size: str, guide: type[PydanticModel], cot_prefill: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    class NoEOSTextStreamer(TextStreamer):
        def __init__(self, tokenizer, logits_processor=None, **kwargs):
            """Streamer is 'released' when the logits processor is 'triggered'."""
            super().__init__(tokenizer, **kwargs)
            self.logits_processor = logits_processor
            self.released = False

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if stream_end:
                eos = self.tokenizer.special_tokens_map["eos_token"]
                text = text.removesuffix(eos)
            lp = self.logits_processor
            if lp.triggered:
                if not self.released:
                    eoc = "</think>"
                    self.released = text.find(eoc) == 0
                    if self.released:
                        text = text.removeprefix(eoc)

                if self.released:
                    print(text, flush=True, end="" if not stream_end else None)

    model_name = f"unsloth/DeepSeek-R1-Distill-Qwen-{model_size}-bnb-4bit"
    logging.getLogger("transformers.utils.quantization_config").setLevel(logging.ERROR)
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logits_processor = load_logits_processor(
        tokenizer=tokenizer, guide=guide, cot_prefill=cot_prefill
    )
    streamer = NoEOSTextStreamer(
        tokenizer, logits_processor=logits_processor[0], skip_prompt=True
    )
    return model, tokenizer, streamer, logits_processor


def load_logits_processor(
    tokenizer: AutoTokenizer, guide: type[PydanticModel], cot_prefill: str
) -> LogitsProcessorList:
    import torch
    from outlines.models import TransformerTokenizer
    from outlines.processors import JSONLogitsProcessor
    from transformers import LogitsProcessor, LogitsProcessorList

    class TriggerBasedLogitsProcessor(LogitsProcessor):
        def __init__(self, tokenizer, base_processor, guide):
            """Logits processor is 'triggered' when it sees the end of CoT token."""
            self.tokenizer = tokenizer
            self.cot_prefill = cot_prefill
            self.internal_monologue = {"reasoning": ""}
            self.base_processor = base_processor
            self.trigger_tokens = (
                tokenizer.encode("</think>", add_special_tokens=False)[0]
                .squeeze(0)
                .tolist()
            )
            self.guide = guide
            self.history = []
            self.triggered_at = -1
            self.triggered = False

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
        ) -> torch.FloatTensor:
            # Add latest token to history
            if len(input_ids.shape) == 2:
                latest_token = input_ids[0, -1].item()
            else:
                latest_token = input_ids[-1].item()
            self.history.append(latest_token)

            # Check if trigger sequence is in recent history
            if len(self.history) >= len(self.trigger_tokens):
                recent = self.history[-len(self.trigger_tokens) :]
                if recent == self.trigger_tokens:
                    self.triggered_at = len(input_ids[0])
                    self.triggered = True
                    # Now recreate the base processor to be a very specific
                    # Hardcode the CoT string constant into the generator
                    cot = self.cot_prefill + self.tokenizer.decode([self.history])[0]
                    new_json_schema = hardcoded_schema(guide=self.guide, cot=cot)
                    self.base_processor = JSONLogitsProcessor(
                        new_json_schema, self.base_processor.tokenizer
                    )

            # Only apply base processor if triggered
            if self.triggered:
                return self.base_processor(input_ids, scores)
            return scores

    outlines_tokenizer = TransformerTokenizer(tokenizer)
    json_schema = json.dumps(guide.model_json_schema())
    guided_processor = JSONLogitsProcessor(json_schema, outlines_tokenizer)
    conditional_guide_processor = TriggerBasedLogitsProcessor(
        tokenizer=outlines_tokenizer, base_processor=guided_processor, guide=guide
    )
    logits_processor = LogitsProcessorList([conditional_guide_processor])
    return logits_processor


def think(
    messages: list[str],
    guide: type[PydanticModel],
    model_size: Literal["1.5B", "7B", "14B", "32B"] = "7B",
    empty_cot: bool = False,
    cot_prefill: str = "<think>\nOkay, so ",
    max_new_tokens: int | None = None,
    deterministic: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    extra_stop_words: list[str] | None = None,
    tee: bool = True,
) -> None | tuple[str, str]:
    model, tokenizer, streamer, logits_processor = load_r1(
        model_size, guide=guide, cot_prefill=cot_prefill
    )
    msg_list = [{"role": "user", "content": msg} for msg in messages]
    prompt_fmt = tokenizer.apply_chat_template(
        msg_list, tokenize=False, add_generation_prompt=True
    )
    if empty_cot:
        prompt_fmt += f"{bot}\n\n{eot}\n"
    else:
        prompt_fmt += cot_prefill
    inputs = tokenizer(prompt_fmt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    terminators = [tokenizer.eos_token_id]
    if extra_stop_words:
        terminators += [tokenizer.convert_tokens_to_ids(w) for w in extra_stop_words]
    # print("\033[32m" + prompt_fmt[: prompt_fmt.find(bot)] + "\033[0m", end="")
    reply = model.generate(
        **inputs,
        streamer=streamer,  # if tee else None,
        max_new_tokens=max_new_tokens,
        do_sample=not deterministic,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=terminators,
        logits_processor=logits_processor,
    )
    cot_end_idx = logits_processor[0].triggered_at
    no_cot = tokenizer.decode(reply[0][cot_end_idx:], skip_special_tokens=True)
    cot = cot_prefill + tokenizer.decode(reply[0][len(inputs[0]) : cot_end_idx])
    if not tee:
        reply_str = tokenizer.decode(reply[0], skip_special_tokens=True)
        return reply_str


class TextAnswer(BaseModel):
    reasoning: list[str]
    answer: str


class NumericAnswer(BaseModel):
    reasoning: list[str]
    answer: int


class DecimalAnswer(BaseModel):
    reasoning: list[str]
    answer: float


def silencio(
    message: str,
    *,
    answer_type: Literal["int", "str", "float"],
    model_size: Literal["1.5B", "7B", "14B", "32B"] = "7B",
    cot_prefill: str = "<think>\nOkay, so ",
    max_new_tokens: int | None = None,
    deterministic: bool = False,
    temperature: float | None = None,
    double_think: bool = True,
    top_p: float | None = None,
):
    message = (
        "Consider the following question which I want you to answer:"
        f"\n\n'''{message}'''\n\n"
        f"Specifically I want you to tell me the answer with a data type of {answer_type}."
    )
    cot_prefill_extra = (
        "the user has asked me to answer their question and I must give "
        f"the answer with the specific data type of {answer_type}."
    )
    model_type = {
        "float": DecimalAnswer,
        "int": NumericAnswer,
        "str": TextAnswer,
    }[answer_type]
    think(
        messages=[message],
        guide=model_type,
        model_size=model_size,
        cot_prefill=cot_prefill + cot_prefill_extra,
        max_new_tokens=max_new_tokens,
        deterministic=deterministic,
        temperature=temperature,
        top_p=top_p,
        tee=True,
    )


def main():
    argh.dispatch_command(silencio, old_name_mapping_policy=False)
