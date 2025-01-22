## Serving with vLLM

### bitsandbytes

```sh
vllm serve "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit" --quantization bitsandbytes --load-format bitsandbytes
```

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
./benchmark/benchmark.py a-helpful-name-for-this-run --model $AIDER_MODEL --edit-format diff --threads 10 --exercises-dir polyglot-benchmark
```

(I might package that behind the same CLI too but for now just use this command...)

### AWQ (recommended)

Alternatively, use Casper Hansen's AWQ quantised model:

```sh
vllm serve "casperhansen/deepseek-r1-distill-qwen-1.5b-awq" --quantization awq --dtype half
```

Then just change the model name to 

```sh
export AIDER_MODEL="hosted_vllm/casperhansen/deepseek-r1-distill-qwen-1.5b-awq"
```

- I also recommend `--gpu-memory-utilization 0.95` to avoid recomputing the KV cache

> **Note**: you can also pass `awq_marlin` as the quantization type, I didn't try this

### GGUF

[nisten](https://x.com/nisten/status/1881419672987541717) (who did a lot of [testing](https://x.com/nisten/status/1874996106540503367)
on the degradation in benchmark perf on DeepSeek v3) recommends GGUF in `q4_k_I` or `q5_k_I`

> the I type leaves the embedding weight and output weight in 8bit which helps out accuracy at
> longer contexts

See: [bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)

I think you would use this like:

```sh
vllm serve ./<model_filename>.gguf --tokenizer bartowski/DeepSeek-R1-Distill-Qwen-14B --quantization gguf
```

### Unquantised

To run the unquantised model:

```sh
vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

and

```sh
export AIDER_MODEL="hosted_vllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

### Concurrency

- 166 threads is possible with vLLM and the AWQ quantised weights (only 225 maximum threads in Aider bench)
- Reaches 1300-1800 TPS average generation throughput for an ETA of ~3-5 hours (whole) or ~20 mins (diff),
  I presume this is an indication that the whole format makes it go off track and just generate endlessly
