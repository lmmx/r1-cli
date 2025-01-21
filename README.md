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

## bitsandbytes

You can then run the [Aider benchmark](https://github.com/Aider-AI/aider/blob/main/benchmark/README.md)
and will be able to reach it with:

```sh
curl -X POST "http://host.docker.internal:8000/v1/chat/completions"   -H "Content-Type: application/json"     --data '{
        "model": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
        "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

and then use in Aider:

```sh
export OPENAI_API_BASE="http://host.docker.internal:8000/v1"
export AIDER_MODEL="hosted_vllm/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit"
./benchmark/benchmark.py a-helpful-name-for-this-run --model $AIDER_MODEL --edit-format whole --threads 10 --exercises-dir polyglot-benchmark
```

(I might package that behind the same CLI too but for now just use this command...)

## AWQ

Alternatively, use Casper Hansen's AWQ quantised model:

```sh
vllm serve "casperhansen/deepseek-r1-distill-qwen-1.5b-awq" --quantization awq --dtype half
```

Then just change the model name to 

```sh
export AIDER_MODEL="hosted_vllm/casperhansen/deepseek-r1-distill-qwen-1.5b-awq"
```
