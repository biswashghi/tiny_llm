import torch


DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def choose_device(requested: str = "auto") -> str:
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"unknown device: {requested}")

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if requested == "mps" and not mps_is_available():
        raise RuntimeError(
            "MPS was requested, but PyTorch cannot use Apple's Metal backend here."
        )

    return requested


def describe_device(device: str) -> str:
    if device == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return "mps (Apple Silicon GPU via Metal, not the Neural Engine)"
    return "cpu"
