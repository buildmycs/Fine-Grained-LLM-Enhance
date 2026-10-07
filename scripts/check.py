python - <<'PY'
from importlib.metadata import version, PackageNotFoundError

for name in ["flashinfer-python", "flashinfer-cubin",
             "flashinfer-jit-cache", "vllm", "torch"]:
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError:
        print(f"{name}: not installed")
