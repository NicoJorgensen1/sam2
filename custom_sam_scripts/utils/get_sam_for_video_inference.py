from AITrainingSharedLibrary.get_relevant_dirs import add_dirs_to_path
from custom_sam_scripts.utils.get_device_to_use import get_device
from custom_sam_scripts.utils.get_sam_checkpoint_and_config import get_sam_checkpoint_and_config
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
) -> None:
    # Get the checkpoint and model config
    checkpoint, model_cfg = get_sam_checkpoint_and_config(model_weights_dir, config_dir, model_size)
    device = get_device()

    # Build the SAM2 video predictor
    if not auto_mask_generator:
        return build_sam2_video_predictor(config_file=str(model_cfg), ckpt_path=checkpoint, device=device)
    
    # Build the SAM2 automatic mask generator
    sam2 = build_sam2(config_file=str(model_cfg), ckpt_path=checkpoint, device=device, apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(sam2)