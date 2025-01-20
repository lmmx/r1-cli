import logging
from typing import Literal

import argh


def load_r1(model_size: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    class NoEOSTextStreamer(TextStreamer):
        def on_finalized_text(self, text: str, stream_end: bool = False):
            if stream_end:
                eos = self.tokenizer.special_tokens_map["eos_token"]
                text = text.removesuffix(eos)
            print(text, flush=True, end="" if not stream_end else None)

    model_name = f"unsloth/DeepSeek-R1-Distill-Qwen-{model_size}-bnb-4bit"
    logging.getLogger("transformers.utils.quantization_config").setLevel(logging.ERROR)
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True)
    return model, tokenizer, streamer


def talk_to_r1(
    messages: list[str],
    model_size: Literal["1.5B", "7B", "14B", "32B"] = "7B",
    max_new_tokens: int | None = None,
    deterministic: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
) -> None:
    model, tokenizer, streamer = load_r1(model_size)
    msg_list = [{"role": "user", "content": msg} for msg in messages]
    prompt_fmt = tokenizer.apply_chat_template(
        msg_list, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_fmt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    _ = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=not deterministic,
        temperature=temperature,
        top_p=top_p,
    )


def main():
    argh.dispatch_command(talk_to_r1)
