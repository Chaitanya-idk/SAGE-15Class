import io
from typing import Any, Optional, Tuple
from PIL import Image, ImageOps


def process_image_safely(
    image_input: Any,
    target_size: Tuple[int, int] = (488, 488),
    preserve_aspect_ratio: bool = True,
    interpolation: str = "bicubic",
    to_rgb: bool = True,
    apply_exif: bool = True
) -> Optional[Image.Image]:
    """Safely loads, rotates by EXIF, converts to RGB, and resizes image preserving aspect ratio.
    
    Returns None if image is corrupt or unreadable.
    """
    try:
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        elif hasattr(image_input, "convert"): # HuggingFace image feature dict or PIL
            image = image_input
        elif isinstance(image_input, dict) and "bytes" in image_input and image_input["bytes"]:
            image = Image.open(io.BytesIO(image_input["bytes"]))
        else:
            return None

        # Verify image integrity
        image.load()

        # Apply EXIF orientation if present
        if apply_exif:
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass

        # Convert to RGB
        if to_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        # Resize preserving aspect ratio with padding
        target_w, target_h = target_size
        interp_method = getattr(Image.Resampling, interpolation.upper(), Image.Resampling.BICUBIC)

        if preserve_aspect_ratio:
            orig_w, orig_h = image.size
            if orig_w == 0 or orig_h == 0:
                return None

            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))

            resized_img = image.resize((new_w, new_h), resample=interp_method)

            # Pad with neutral background (0,0,0) or (128,128,128)
            padded_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            pad_x = (target_w - new_w) // 2
            pad_y = (target_h - new_h) // 2
            padded_img.paste(resized_img, (pad_x, pad_y))
            return padded_img
        else:
            return image.resize((target_w, target_h), resample=interp_method)

    except Exception:
        return None
