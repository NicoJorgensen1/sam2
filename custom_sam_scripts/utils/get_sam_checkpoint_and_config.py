from CameraAISharedLibrary.recursive_search import search_files
from typing import Tuple
from pathlib import Path
import os




def get_sam_checkpoint_and_config(
    model_weights_dir: str,
    config_dir: str, model_size: str,
    return_relative_config_path: bool = True,
) -> Tuple[str, Path]:
        # Find SAM2 weights
        model_paths = search_files(start_path=model_weights_dir, accepted_img_extensions=(".pt", ".pth"))
        sam_weights = [model_path for model_path in model_paths if "sam2" in model_path.lower() and model_size.lower() in model_path.lower()]
        if not sam_weights:
            raise ValueError(f"No SAM2 weights found for model size {model_size} in {model_weights_dir}")
        if len(sam_weights) > 1:
            raise ValueError(f"Found multiple SAM2 weights for model size {model_size} in {model_weights_dir}")
        checkpoint = sam_weights[0]
        
        # Find the model config checkpoint 
        model_cfg_list = [Path(config_path) for config_path in search_files(start_path=config_dir, accepted_img_extensions=(".yaml", ".yml"))]
        model_cfg_list = [model_cfg for model_cfg in model_cfg_list if "sam2.1" in str(model_cfg).lower() and model_size.lower()[0] in model_cfg.stem.lower()]
        if not model_cfg_list:
            raise ValueError(f"No SAM2 config found for model size {model_size} in {config_dir}")
        if len(model_cfg_list) > 1:
            raise ValueError(f"Found multiple SAM2 configs for model size {model_size} in {config_dir}")
        model_cfg = model_cfg_list[0]

        # Hydra config (which loads the model config) expects relative paths - for some reason without the relative path root
        if return_relative_config_path:
            relative_path = os.path.relpath(model_cfg, os.getcwd())
            model_cfg = os.path.join(*relative_path.split(os.sep)[1:])

        return checkpoint, model_cfg