# Installation

Use Linux, Python 3.11, and an NVIDIA GPU for training. The shared environment pins
PyTorch 2.9.1, torchvision 0.24.1, and CUDA 12.8. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
bash setup/bootstrap.sh
source .venv/bin/activate
```

Set `LAPLACIAN_PYTHON` to a Python 3.11 interpreter if needed. CUDA libraries come
with the PyTorch wheels; the NVIDIA driver must support CUDA 12.8. Language FP8
training requires H100-class hardware. Plotting and small analysis runs also work
on CPU.

- `setup/bootstrap.sh`: creates the shared environment and sets up all three tasks.
- `setup/requirements.lock`: pinned runtime and build dependencies.
- `setup/requirements-extra.in`: shared dependency constraints used to generate the lock.

Activate this environment before running commands from a task directory. Use the
shared bootstrap for all three tasks. Dataset layouts and training/measurement
commands are in each task README. Write generated results under `outputs/`.
W&B tracking is optional.
