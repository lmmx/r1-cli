## Serving with TGI

We'll use the same port (8000) as used for vLLM.

```sh
model=casperhansen/deepseek-r1-distill-qwen-14b-awq
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8000:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id $model
```

- Note on Docker GPU setup
  [here](https://github.com/lmmx/devnotes/wiki/Run-a-docker-container-with-GPU)

## Aider

For Aider with TGI note that the underlying LiteLLM will not use `hosted_vllm` route prefix,
but it's unclear which other prefix to use for TGI. Using the `hosted_vllm` one works!

As before we expor the `OPENAI_API_BASE` for Aider to pick up for benchmarks from within the
benchmark Docker container

```sh
export OPENAI_API_BASE="http://host.docker.internal:8000/v1"
export AIDER_MODEL="hosted_vllm/casperhansen/deepseek-r1-distill-qwen-14b-awq"

./benchmark/benchmark.py bench-14b-awq-py-diff --model $AIDER_MODEL \
   --edit-format diff --threads 2000 --exercises-dir polyglot-benchmark \
   --new --num-tests -1 --languages python
```

## 14B

Time per token is reported with peaks of roughly 50ms equating to a tokens per second (TPS) of 20
tokens per second (this is per thread, so multiply that by the number of parallel processes - this
is not clear).

I also tried the 8-bit precision 14B from Knut Jaegersberg:

```sh
model=KnutJaegersberg/DeepSeek-R1-Distill-Qwen-14B-exl2-8.0bpw
```
