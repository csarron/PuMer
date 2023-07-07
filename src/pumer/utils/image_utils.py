import io

from PIL import Image
from torchvision import transforms as T

from .randaug import RandAugment

# Image.MAX_IMAGE_PIXELS = None


def get_image_transforms(image_size=384, use_randaug=False, is_clip=False):
    transformations = []
    if use_randaug:
        transformations.append(RandAugment(2, 9))

    if is_clip:
        interpolation = T.InterpolationMode.BICUBIC
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        interpolation = T.InterpolationMode.BILINEAR
        # inception style normalize
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    transformations.append(T.Resize([image_size, image_size], interpolation=interpolation))
    if is_clip:
        transformations.append(T.CenterCrop(image_size))
    transformations.extend(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return T.Compose(transformations)


def image_bytes_to_pil(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def image_file_to_pil(image_file):
    return Image.open(image_file).convert("RGB")
