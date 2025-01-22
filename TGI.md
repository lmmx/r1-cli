## Serving with TGI

```sh
model=casperhansen/deepseek-r1-distill-qwen-14b-awq
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id $model
```

(Paused to update my drivers)
