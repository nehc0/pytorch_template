# PyTorch template

A neat and informative PyTorch template, implementing many useful features, has wide applicability.

Examples showing how to customize this template for a real project:

- [`MNIST`](./example_mnist/)
- [`SST2`](./example_sst2/)

## Features

- utilize [PyTorch DDP](https://pytorch.org/tutorials/beginner/ddp_series_intro.html)
to enable training across multiple GPUs and nodes,
only tweak [`./run.sh`](./run.sh) to fit different architectures
- clear workflow in [`./main.py`](./main.py)
- if loading dataset from [Hugging Face](https://huggingface.co/datasets),
minimal code adjustments are needed
- implement basic preprocess for cv and nlp in [`./preprocess.py`](./preprocess.py)
- an all-in-one **Trainer** off the shelf in [`./trainer.py`](./trainer.py),
which works well with [wandb](https://wandb.ai/site)
- well-structured config in [`./config.yaml`](./config.yaml)
- some utility functions and classes in [`./utils.py`](./utils.py)
- simple `sh ./run.sh` to launch
