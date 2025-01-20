import logging

import argh
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def load_r1():
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit"
    logging.getLogger("transformers.utils.quantization_config").setLevel(logging.ERROR)
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    return model, tokenizer, streamer


def talk_to_r1(
    messages: list[str],
    max_new_tokens: int | None = None,
    deterministic: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
) -> None:
    model, tokenizer, streamer = load_r1()
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
