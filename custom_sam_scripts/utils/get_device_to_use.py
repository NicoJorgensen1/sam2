import torch



def get_device():
    """
    Determine and configure the appropriate computational device for PyTorch operations.

    This function checks for available hardware and returns the most suitable device
    for running PyTorch operations. It also applies specific optimizations for CUDA devices.

    Returns:
        torch.device: The selected device (cuda, mps, or cpu).

    Side effects:
        - For CUDA devices:
            - Enables bfloat16 autocast.
            - Enables TensorFloat-32 (TF32) for Ampere GPUs and later.
        - For MPS devices:
            - Prints a warning about preliminary support and potential performance issues.

    Note:
        The function prioritizes CUDA > MPS > CPU in device selection.
    """
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

    return device



### Run from CLI 
if __name__ == "__main__":
    device = get_device()
    print(f"The device used is '{device}'")
