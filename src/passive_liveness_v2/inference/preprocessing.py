"""Image decoding and preprocessing.

Mirrors the training pipeline (``PADFrameDataset`` in ``notebook.py``):
Resize((224, 224)) -> ToTensor -> ImageNet normalization. Extraction-time
and serve-time preprocessing must stay numerically consistent.
"""

from io import BytesIO

import PIL.Image
from torchvision import transforms

from passive_liveness_v2.core.errors import APIError

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(target_size: int) -> transforms.Compose:
    """Build the train/serve preprocessing pipeline for a given input size."""
    return transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def decode_image(data: bytes) -> PIL.Image.Image:
    """Decode raw upload bytes into an RGB PIL image.

    Raises:
        APIError: if the payload is not a decodable image.
    """
    try:
        image = PIL.Image.open(BytesIO(data))
        image.load()  # force a full decode; raises on truncated/corrupt files
    except Exception as exc:
        raise APIError(
            "invalid_image",
            "Uploaded file is not a decodable image",
            status_code=400,
            details={"reason": str(exc)},
        ) from exc
    return image.convert("RGB")
