# r1

Simple app providing a CLI command to run DeepSeek r1 (4 bit quant of the Qwen 32B distilled model,
via unsloth and vLLM)

## Installation

I recommend using `uv` and running `compileall` on the `.venv` directory to get 40% faster import times:

```sh
uv venv
source .venv/bin/activate
uv pip install -e .
$(uv python find -m compileall .)
``` 

You can then symlink your entrypoint at `.venv/bin/r1` to a directory in your path for use as a command without activating the venv,

```sh
~/opt/bin $ ln -s /home/louis/lab/r1/.venv/bin/r1 r1
```

or else export it onto your PATH in your bashrc.

## Usage

Pass prompt messages (which will be given role of user) via CLI, as well as flags such as
`--temperature` (default 1.0 is good, up to 1.8 can work, above that it loses coherence):

```sh
usage: r1 [-h] [-m [MAX_NEW_TOKENS]] [-d] [--temperature [TEMPERATURE]]
          [--top-p [TOP_P]]
          [messages ...]

positional arguments:
  messages              -

options:
  -h, --help            show this help message and exit
  -m [MAX_NEW_TOKENS], --max-new-tokens [MAX_NEW_TOKENS]
                        -
  -d, --deterministic   False
  --temperature [TEMPERATURE]
                        -
  --top-p [TOP_P]       -
```

## Serving

You can serve with

```sh
vllm serve "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit" --quantization bitsandbytes --load-format bitsandbytes
```

## Aider benchmark

There are 2 quantisation formats on HuggingFace for r1: bitsandbytes and AWQ

## Serving with vLLM

See [VLLM](https://github.com/lmmx/r1-cli/blob/master/VLLM.md) notes.

## Serving with TGI

See [TGI](https://github.com/lmmx/r1-cli/blob/master/VLLM.md) notes.

## Python benchmarking

Aider bench, Python subset, 4-bit AWQ on vLLM: https://gist.github.com/lmmx/ab6563e681d936fd9c3c864447fbf19f

- 1.5B ⇢ 0% pass@1, 0% pass@2, 35% well-formed
- 7B ⇢ 0% pass rate, 42% well-formed

Aider bench, Python subset, 4-bit AWQ on TGI

- 14B ⇢ ?% pass rate, ?% well-formed
- 32B ⇢ ?% pass rate, ?% well-formed
