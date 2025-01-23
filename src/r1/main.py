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


def think(
    messages: list[str],
    model_size: Literal["1.5B", "7B", "14B", "32B"] = "7B",
    empty_cot: bool = False,
    cot_prefill: str = "<think>\nOkay, so ",
    max_new_tokens: int | None = None,
    deterministic: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    tee: bool = True,
) -> None | tuple[str, str]:
    model, tokenizer, streamer = load_r1(model_size)
    msg_list = [{"role": "user", "content": msg} for msg in messages]
    prompt_fmt = tokenizer.apply_chat_template(
        msg_list, tokenize=False, add_generation_prompt=True
    )
    if empty_cot:
        prompt_fmt += "<think>\n\n</think>\n"
    else:
        prompt_fmt += cot_prefill
    inputs = tokenizer(prompt_fmt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    reply = model.generate(
        **inputs,
        streamer=streamer,  # if tee else None,
        max_new_tokens=max_new_tokens,
        do_sample=not deterministic,
        temperature=temperature,
        top_p=top_p,
    )
    if not tee:
        reply_str = tokenizer.decode(reply[0], skip_special_tokens=True)
        prompt, think_and_answer = reply_str.split("<think>", 1)
        cot, answer = think_and_answer.split("</think>", 1)
        return prompt, cot, answer


def wish(
    messages: list[str],
    analysis: str = "what this message portrays about my character, its revealed/manifest motives are, and what you think of these motives",
    model_size: Literal["1.5B", "7B", "14B", "32B"] = "7B",
    cot_prefill: str = "<think>\nOkay, so ",
    max_new_tokens: int | None = None,
    deterministic: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
):
    user_command = "\n".join(messages)
    message = (
        "Consider the following task I want to do. The task request says:"
        f"\n\n'''{user_command}'''\n\n"
        "I am contacting you to consider this task. "
        "I would like you to consider this message objectively and what it demonstrates about my character"
        f"Specifically I want you to tell me {analysis}."
    )
    cot_prefill_extra = "the user has asked me to review a command and first I will consider what carrying out the action would demonstrate about their character"
    prompt, cot, answer = think(
        messages=[message],
        model_size=model_size,
        cot_prefill=cot_prefill + cot_prefill_extra,
        max_new_tokens=max_new_tokens,
        deterministic=deterministic,
        temperature=temperature,
        top_p=top_p,
        tee=False,
    )
    exec_message = (
        "I have been planning to "
        f"{user_command}"
        f".\n\nI got an AI to assess the motives and empirical character demonstrated by this idea, which gave the following:\n\n'''\n{cot}\n'''\n\n"
        "\n\nAfter consideration of this, I decided I do indeed wish to carry it out and was hoping you can help me on the execution."
    )
    exec_cot = cot_prefill
    think(
        messages=[exec_message],
        model_size=model_size,
        cot_prefill=exec_cot,
        max_new_tokens=max_new_tokens,
        deterministic=deterministic,
        temperature=temperature,
        top_p=top_p,
        tee=True,
    )


def main():
    argh.dispatch_commands([think, wish])
