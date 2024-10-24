from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from custom_sam_scripts.utils.general_utils.get_device_to_use import get_device
from custom_sam_scripts.utils.sam2_utils.get_sam_checkpoint_and_config import get_sam_checkpoint_and_config
from typing import Union, Any
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
import os

relevant_dirs = add_dirs_to_path()




def get_sam_for_video_inference(
        model_size: str = "large",
        model_weights_dir: str = relevant_dirs.get("model_weights_dir", os.getenv("MODEL_WEIGHTS_DIR", "checkpoints")),
        config_dir: str = os.path.join(os.getcwd(), "sam2", "configs"),
        auto_mask_generator: bool = True,
) -> Union[SAM2AutomaticMaskGenerator, Any]:
    """
    Initialize and return a SAM2 (Segment Anything Model 2) model for video inference.

    This function sets up either an automatic mask generator or a video predictor
    based on the specified parameters and available resources.

    Args:
        model_size (str, optional): Size of the SAM2 model to use. Defaults to "large".
        model_weights_dir (str, optional): Directory containing the model weights.
            Defaults to a path from relevant_dirs or an environment variable.
        config_dir (str, optional): Directory containing the model configuration files.
            Defaults to "sam2/configs" in the current working directory.
        auto_mask_generator (bool, optional): If True, returns an automatic mask generator.
            If False, returns a video predictor. Defaults to True.

    Returns:
        Union[SAM2AutomaticMaskGenerator, Any]: Either a SAM2AutomaticMaskGenerator instance
        if auto_mask_generator is True, or a SAM2 video predictor instance otherwise.

    Note:
        This function relies on several utility functions and external dependencies:
        - get_sam_checkpoint_and_config: To locate model checkpoints and configs.
        - get_device: To determine the appropriate computational device.
        - build_sam2_video_predictor or build_sam2: To construct the SAM2 model.

    The function automatically selects the appropriate device (CUDA, MPS, or CPU)
    for model inference based on availability.
    """
    # Get the checkpoint and model config
    checkpoint, model_cfg = get_sam_checkpoint_and_config(model_weights_dir, config_dir, model_size)
    device = get_device()

    # Build the SAM2 video predictor
    if not auto_mask_generator:
        return build_sam2_video_predictor(config_file=str(model_cfg), ckpt_path=checkpoint, device=device)
    
    # Build the SAM2 automatic mask generator
    sam2 = build_sam2(config_file=str(model_cfg), ckpt_path=checkpoint, device=device, apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(sam2)
