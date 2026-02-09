import numpy as np
from PIL import Image

def center_crop_arr(pil_image, image_size):
    # when pil_image is huge, do repeated halving
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample = Image.Box    # fast downsampling, good for integer factors
        )

    # ensures shorter side of pil_image becomes exactly image_size
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample = Image.BICUBIC    # high-quality resizing, better for non-integer scaling
    )

    # converts image to array: arr.shape -> (height, width, channels)
    arr = np.array(pil_image)
    # compute center crop coordinates by finding top-left corner of the centered crop
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    # return a center-cropped square PIL image of exactly image_size × image_size
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
