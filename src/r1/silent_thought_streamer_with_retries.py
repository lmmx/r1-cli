from __future__ import annotations

import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypeVar

import argh
from pydantic import BaseModel, create_model
from pysnooper import snoop

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

bot, eot = "<think>", "</think>"


@snoop()
def load_r1(model_size: str, guide: type[PydanticModel], cot_prefill: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    class NoEOSTextStreamer(TextStreamer):
        def __init__(self, tokenizer, logits_processor=None, **kwargs):
            """Streamer is 'post-CoT' when the logits processor is 'triggered'."""
            super().__init__(tokenizer, **kwargs)
            self.logits_processor = logits_processor
            self.post_cot = False
            cot_prefill_cleaned = cot_prefill.removeprefix(bot).lstrip()
            self.init_json = json.dumps({"reasoning": cot_prefill_cleaned})[:-2]

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if stream_end:
                eos = self.tokenizer.special_tokens_map["eos_token"]
                text = text.removesuffix(eos)
            lp = self.logits_processor
            if lp.triggered:
                if not self.post_cot:
                    self.post_cot = text.find(eot) == 0
                    if self.post_cot:
                        text = text.replace(eot, '", ', 1).replace("{", "", 1)

            text = text if self.post_cot else json.dumps(text)[1:-1]
            if self.init_json:
                text = self.init_json + text
                self.init_json = ""
            print(text, flush=True, end="" if not stream_end else None)
            self.logits_processor.result += text

    model_name = f"unsloth/DeepSeek-R1-Distill-Qwen-{model_size}-bnb-4bit"
    logging.getLogger("transformers.utils.quantization_config").setLevel(logging.ERROR)

    # stored = Path(__file__).parent / f"{model_name.replace('/', '--')}.pt"
    # if stored.exists():
    #     model, tokenizer = torch.load(stored, map_location="cuda", weights_only=False)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     torch.save((model, tokenizer), stored)
    # print(model.device)
    model, tokenizer = cache.load(model_name, model_cls="AutoModelForCausalLM")

    logits_processors = load_logits_processors(
        tokenizer=tokenizer, guide=guide, cot_prefill=cot_prefill
    )
    trigger_lp = logits_processors[0]
    streamer = NoEOSTextStreamer(
        tokenizer, logits_processor=trigger_lp, skip_prompt=True
    )
    return model, tokenizer, streamer, logits_processors


def load_logits_processors(
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
            self.base_processor = base_processor
            self.trigger_tok = tokenizer.encode(eot, add_special_tokens=False)[0].item()
            self.guide = guide
            self.history = []
            self.result = ""
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
            recent = self.history[-1]
            if recent == self.trigger_tok:
                self.triggered_at = len(input_ids[0])
                self.triggered = True

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
    logits_processors = LogitsProcessorList([conditional_guide_processor])
    return logits_processors


@snoop()
def think(
    messages: list[str],
    guide: type[PydanticModel],
    model_size: Literal["1.5B", "7B", "14B", "32B"] = "7B",
    empty_cot: bool = False,
    cot_prefill: str = "\nOkay, so ",
    max_new_tokens: int | None = None,
    deterministic: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    extra_stop_words: list[str] | None = None,
    tee: bool = True,
) -> None | tuple[str, str]:
    model, tokenizer, streamer, logits_processors = load_r1(
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
        logits_processor=logits_processors,
    )
    # cot_end_idx = trigger_lp.triggered_at
    # no_cot = tokenizer.decode(reply[0][cot_end_idx:], skip_special_tokens=True)
    # cot = cot_prefill + tokenizer.decode(reply[0][len(inputs[0]) : cot_end_idx])
    # real_generation = tokenizer.decode(reply[0], skip_special_tokens=True)
    trigger_lp = logits_processors[0]
    json_generation = trigger_lp.result
    return json_generation


class TextAnswer(BaseModel):
    # reasoning: list[str]
    answer: str


class NumericAnswer(BaseModel):
    # reasoning: list[str]
    answer: int


class DecimalAnswer(BaseModel):
    # reasoning: list[str]
    answer: float


def silencio(
    message: str,
    *,
    answer_type: Literal["int", "str", "float"],
    model_size: Literal["1.5B", "7B", "14B", "32B"] = "7B",
    cot_prefill: str = "\nOkay, so ",
    max_new_tokens: int | None = None,
    deterministic: bool = False,
    temperature: float | None = None,
    double_think: bool = True,
    top_p: float | None = None,
    return_result: bool = False,
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
    # String result of the text that was streamed
    result = think(
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
    if return_result:
        return result


def main():
    argh.dispatch_command(silencio, old_name_mapping_policy=False)
