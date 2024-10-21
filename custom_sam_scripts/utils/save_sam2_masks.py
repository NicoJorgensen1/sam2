from typing import Optional
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import logging




def save_results(video_segments, output_dir, logger: Optional[logging.Logger] = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, obj_masks in tqdm(video_segments.items(), desc="Saving results", total=len(video_segments), leave=False):
        for obj_id, mask in obj_masks.items():
            # Convert binary mask to PIL Image
            mask_image = Image.fromarray((mask.squeeze() * 255).astype(np.uint8))
            # Save as PNG
            mask_image.save(output_dir / f"frame_{frame_idx}_object_{obj_id}.png")
    if logger:
        logger.info(f"Results saved to {output_dir}")
