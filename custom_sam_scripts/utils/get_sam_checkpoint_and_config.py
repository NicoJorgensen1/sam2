from CameraAISharedLibrary.recursive_search import search_files
from typing import Tuple
from pathlib import Path
import os




def get_sam_checkpoint_and_config(
    model_weights_dir: str,
    config_dir: str,
    model_size: str,
    return_relative_config_path: bool = True,
) -> Tuple[str, Path]:
    """
    Locate and return the SAM2 model checkpoint and configuration file paths.

    This function searches for SAM2 model weights and configuration files based on the
    specified model size and directories. It ensures that exactly one matching checkpoint
    and one matching configuration file are found.

    Args:
        model_weights_dir (str): Directory containing SAM2 model weight files.
        config_dir (str): Directory containing SAM2 configuration files.
        model_size (str): Size of the SAM2 model (e.g., "large", "base").
        return_relative_config_path (bool, optional): If True, returns the relative path
            for the config file. Defaults to True.

    Returns:
        Tuple[str, Path]: A tuple containing:
            - str: Path to the SAM2 model checkpoint file.
            - Path: Path to the SAM2 model configuration file.

    Raises:
        ValueError: If no matching checkpoint or config file is found, or if multiple
                    matching files are found.

    Note:
        The function uses case-insensitive matching for file names.
        For configuration files, it looks for files containing "sam2.1" and the first
        letter of the model size in their names.
    """
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
