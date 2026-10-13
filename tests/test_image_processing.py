import io
from PIL import Image
try:
    import pytest
except ImportError:
    pytest = None
from sage.data.image_validation import process_image_safely


def test_process_image_safely_rgb_and_target_size():
    # Create non-square test image (300 x 200, RGBA)
    test_img = Image.new("RGBA", (300, 200), (255, 0, 0, 128))
    processed = process_image_safely(test_img, target_size=(488, 488), preserve_aspect_ratio=True)

    assert processed is not None
    assert processed.size == (488, 488)
    assert processed.mode == "RGB"


def test_process_image_safely_bytes_input():
    img = Image.new("RGB", (100, 100), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    processed = process_image_safely(jpeg_bytes, target_size=(488, 488))
    assert processed is not None
    assert processed.size == (488, 488)
    assert processed.mode == "RGB"


def test_process_image_safely_corrupt_input():
    corrupt_bytes = b"not_an_image_random_corrupted_data_12345"
    processed = process_image_safely(corrupt_bytes)
    assert processed is None
