import glob
import os

from image_fragment.fragment import ImageFragment

# FOR .jpg, .png, .jpeg
from imageio import imread, imsave

# FOR .tiff
from tifffile import imread

ORIGINAL_DIM_OF_IMAGE = (15899, 21817, 3)
CROP_TO_DIM = (384, 384, 3)

image_fragment = ImageFragment.image_fragment_3d(
    fragment_size=(384, 384, 3), org_size=ORIGINAL_DIM_OF_IMAGE
)

IMAGE_DIR = r"/home/ippws/building-footprint-segmentation/data/rio/treatedds/labels"
SAVE_DIR = r"/home/ippws/building-footprint-segmentation/data/rio/cropedds/labelsbu"

for file in glob.glob(
    os.path.join(IMAGE_DIR, "*")
):
    image = imread(file)
    for i, fragment in enumerate(image_fragment):
        # GET DATA THAT BELONGS TO THE FRAGMENT
        fragmented_image = fragment.get_fragment_data(image)

        imsave(
            os.path.join(
                SAVE_DIR,
                f"{i}_{os.path.basename(file)}",
            ),
            fragmented_image,
        )
