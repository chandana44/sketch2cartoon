from __future__ import print_function
from os import listdir
from os.path import isfile, join
import sys

from PIL import Image


if len(sys.argv) != 3:
    print('Please provide <raw_images_path> and <cropped_images_path> as arguments..')
    sys.exit(1)

raw_images_path = sys.argv[1]
cropped_images_path = sys.argv[2]

raw_images_paths = [f for f in listdir(raw_images_path) if isfile(join(raw_images_path, f))]
for image_path in raw_images_paths:
    print(image_path)
    with open(raw_images_path + image_path, 'r+b') as f:
        with Image.open(f) as image:
            image = image.resize((256, 256), Image.ANTIALIAS)
            image.save(cropped_images_path + image_path, image.format)
            #cover = resizeimage.resize_cover(image, [256, 256])
            #cover.save(cropped_images_path + image_path, image.format)
