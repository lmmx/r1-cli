# r1

Simple app providing a CLI command to run DeepSeek r1 (4 bit quant of the Qwen 32B distilled model,
via unsloth)

## Installation

I recommend using `uv` and running `compileall` on the `.venv` directory to get 40% faster import times:

```sh
uv venv
source .venv/bin/activate
uv pip install -e .
$(uv python find -m compileall .)
``` 

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
