from PIL import Image, ImageEnhance
from pylab import *
from scipy.ndimage import filters
from skimage import io
import glob, os


def generate_sketches(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for files1 in glob.glob(in_dir + '/*.png'):
        filepath, filename = os.path.split(files1)
        Gamma = 0.98
        Phi = 200
        Epsilon = 0.1
        k = 2
        Sigma = 1.5

        im = Image.open(files1).convert('L')
        im = array(ImageEnhance.Sharpness(im).enhance(3.0))
        im2 = filters.gaussian_filter(im, Sigma)
        im3 = filters.gaussian_filter(im, Sigma * k)
        differencedIm2 = im2 - (Gamma * im3)
        (x, y) = shape(im2)
        for i in range(x):
            for j in range(y):
                if differencedIm2[i, j] < Epsilon:
                    differencedIm2[i, j] = 1
                else:
                    differencedIm2[i, j] = 250 + tanh(Phi * (differencedIm2[i, j]))

        io.imsave(os.path.join(out_dir, filename), differencedIm2.astype(np.uint8))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Please provide <input_photos_directory> and <output_sketches_directory> as arguments..')
        sys.exit(1)

    generate_sketches(sys.argv[1], sys.argv[2])
