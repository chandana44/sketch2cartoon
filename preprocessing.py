from os import listdir
from os.path import isfile, join
from PIL import Image
from resizeimage import resizeimage

raw_images_path = "minions/"
cropped_images_path = "cartoons/"

raw_images_paths = [f for f in listdir(raw_images_path) if isfile(join(raw_images_path, f))]
for image_path in raw_images_paths:
    print image_path
    with open(raw_images_path + image_path, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [256, 256])
            cover.save(cropped_images_path + image_path, image.format)

