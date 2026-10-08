#!/usr/bin/env python3
"""
CLI wrapper for SAM3 segmentation.
Runs in a separate conda environment (sam3) to avoid dependency conflicts.
"""

import argparse
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="SAM3 text-prompt segmentation")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", required=True, help="Text prompt for segmentation")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--out-mask", required=True, help="Output mask path")
    parser.add_argument("--out-overlay", required=True, help="Output overlay path")
    parser.add_argument("--sam3-repo", default="../sam3", help="Path to SAM3 repository")
    parser.add_argument("--hf-cache-dir", default="/data/huggingface_cache/hub/models--facebook--sam3", help="HuggingFace cache directory")
    
    args = parser.parse_args()
    
    # Add SAM3 to path
    sam3_path = Path(args.sam3_repo).resolve()
    if not sam3_path.exists():
        print(f"Error: SAM3 repository not found at {sam3_path}", file=sys.stderr)
        sys.exit(1)
    
    sys.path.insert(0, str(sam3_path))
    
    # Set HuggingFace cache
    # args.hf_cache_dir should be the HF hub cache root (e.g. /data/huggingface_cache/hub)
    hf_cache_dir = str(Path(args.hf_cache_dir).resolve())
    # HF_HOME typically points to ~/.cache/huggingface; we set it to the parent of hub
    os.environ["HF_HOME"] = str(Path(hf_cache_dir).parent)
    os.environ["HF_HUB_CACHE"] = hf_cache_dir
    
    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam3_image_model(device=device, hf_cache_dir=hf_cache_dir)
        processor = Sam3Processor(model, device=device, confidence_threshold=float(args.confidence))
        
        # Load image
        image = Image.open(args.image).convert("RGB")
        image_array = np.array(image, dtype=np.uint8)
        
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.prompt)
        masks, scores = output.get("masks"), output.get("scores")
        if masks is None or scores is None:
            print("No outputs returned from SAM3.", file=sys.stderr)
            sys.exit(2)

        scores_np = scores.detach().float().cpu().numpy()
        if scores_np.size == 0:
            # No detection above threshold: write an empty mask + passthrough overlay for UI stability
            empty = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
            Image.fromarray(empty, mode="L").save(args.out_mask)
            Image.fromarray(image_array, mode="RGB").save(args.out_overlay)
            print("No masks above confidence threshold.", file=sys.stderr)
            sys.exit(0)

        best_idx = int(scores_np.argmax())
        mask_bool = masks[best_idx].squeeze(0).detach().cpu().numpy().astype(bool)  # (H, W)
        mask = (mask_bool.astype(np.uint8) * 255)
        
        # Save mask
        mask_image = Image.fromarray(mask, mode="L")
        mask_image.save(args.out_mask)
        
        # Create overlay
        overlay = image_array.copy().astype(np.float32)
        overlay[mask_bool] = 0.7 * overlay[mask_bool] + 0.3 * np.array([0, 255, 0], dtype=np.float32)
        overlay_image = Image.fromarray(overlay.clip(0, 255).astype(np.uint8), mode="RGB")
        overlay_image.save(args.out_overlay)
        
        print(f"Mask saved to {args.out_mask}")
        print(f"Overlay saved to {args.out_overlay}")
        
    except Exception as e:
        print(f"Error during SAM3 segmentation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
