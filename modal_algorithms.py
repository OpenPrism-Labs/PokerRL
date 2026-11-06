"""Bounded ephemeral Modal GPU pilot and optional complete algorithm benchmark."""

import io
import json
import zipfile
from pathlib import Path

import modal

app = modal.App("pokerrl-algorithm-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy==2.5.2", "pokerkit==0.7.7", "torch==2.10.0")
    .add_local_python_source("pokerrl")
    .add_local_dir("results/holdem", remote_path="/root/results/holdem")
)


@app.function(image=image, gpu="L4", cpu=4, timeout=900)
def gpu_pilot() -> str:
    """Measure real collection and learning, not only synthetic matrix throughput."""
    import tempfile
    from dataclasses import asdict

    import torch

    from pokerrl.algorithm_experiment import Algorithm, train_one

    if not torch.cuda.is_available():
        raise RuntimeError("Modal GPU function has no CUDA device")
    torch.set_num_threads(1)
    # Initialize kernels before timing full-hand training.
    torch.ones(32, device="cuda").sum().item()
    torch.cuda.synchronize()
    algorithms: tuple[Algorithm, ...] = ("ppo", "ddqn")
    results = []
    with tempfile.TemporaryDirectory(prefix="pokerrl-pilot-") as directory:
        for algorithm in algorithms:
            for device in ("cpu", "cuda"):
                result = train_one(algorithm, 91, 500, device, Path(directory) / device)
                results.append(asdict(result))
    return json.dumps(
        {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cpu_allocation": 4,
            "torch_threads": 1,
            "runs": results,
            "scope": "Single pilot seed; includes simulator, learning and checkpoint serialization",
        },
        indent=2,
    )


@app.function(image=image, gpu="L4", cpu=4, timeout=3600)
def gpu_benchmark(train_hands: int = 3000) -> bytes:
    """Run the declared full experiment and return its owned artifacts."""
    import tempfile

    from pokerrl.algorithm_experiment import run

    with tempfile.TemporaryDirectory(prefix="pokerrl-benchmark-") as directory:
        output = Path(directory)
        run(output, train_hands=train_hands, device="cuda")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in output.iterdir():
                if path.is_file():
                    archive.write(path, arcname=path.name)
        return buffer.getvalue()


@app.local_entrypoint()
def main(
    pilot: bool = False,
    output: str = "results/algorithms",
    train_hands: int = 3000,
) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if pilot:
        result = gpu_pilot.remote()
        (destination / "modal-pilot.json").write_text(result + "\n")
        print(result)
    else:
        if train_hands < 1:
            raise ValueError("Training budget must be positive")
        data = gpu_benchmark.remote(train_hands)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                if Path(name).name != name:
                    raise ValueError("Unexpected non-flat benchmark artifact")
                (destination / name).write_bytes(archive.read(name))
        print(f"Retrieved {len(data)} compressed bytes to {destination}")
