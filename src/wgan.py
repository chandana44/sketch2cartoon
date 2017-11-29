from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

import matplotlib.pyplot as plt

from util import *
import sys

import numpy as np

K.set_image_dim_ordering('th')

img_rows = 64
img_cols = 64
IN_CH = 3
OUT_CH = 3
LAMBDA = 100
NF = 64  # number of filter
BATCH_SIZE = 128

YEARBOOK_TEST_PHOTOS_SAMPLE_PATH = '../data/yearbook_test_photos_sample'


class WGAN():
    def __init__(self, load_weights, generator_checkpoint_file, discriminator_checkpoint_file):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = IN_CH

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        if load_weights == 1:
            self.discriminator.load_weights(discriminator_checkpoint_file)
        self.discriminator.compile(loss=self.wasserstein_loss,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        if load_weights == 1:
            self.generator.load_weights(generator_checkpoint_file)
        self.generator.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_shape=noise_shape))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.summary()

        img = Input(shape=img_shape)
        features = model(img)
        valid = Dense(1, activation="linear")(features)

        return Model(img, valid)

    def train(self, num_epochs, initial_epoch, train_photos_dir, train_sketches_dir, output_dir,
              generator_checkpoint_file, discriminator_checkpoint_file,
              save_image_frequency=50):
        photos = glob.glob(os.path.join(train_photos_dir, '*.png'))
        sketches = glob.glob(os.path.join(train_sketches_dir, '*.png'))

        gen_iterations = 0

        for epoch in range(initial_epoch, num_epochs):
            index = 0
            print(get_time_string() + " Epoch is", epoch)
            print(get_time_string() + " Number of batches", int(len(photos) / BATCH_SIZE))

            for X_train, Y_train in chunks(photos, sketches, BATCH_SIZE):
                print(get_time_string() + ' Batch number: ' + str(index))

                X_train = (X_train.astype(np.float32) - 127.5) / 127.5
                Y_train = (Y_train.astype(np.float32) - 127.5) / 127.5

                gen_iterations += 1

                if gen_iterations <= 225 or gen_iterations % 2500 == 0:
                    self.n_critic = 1
                else:
                    self.n_critic = 5

                half_batch = BATCH_SIZE / 2

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                true_imgs = Y_train[idx]

                noise = np.random.normal(0, 1, (half_batch, 100))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(true_imgs, -np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip discriminator weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

                # ---------------------
                #  Train Generator
                # ---------------------
                if gen_iterations % self.n_critic == 0:
                    noise = np.random.normal(0, 1, (BATCH_SIZE, 100))

                    # Train the generator
                    g_loss = self.combined.train_on_batch(noise, -np.ones((BATCH_SIZE, 1)))

                    # Plot the progress
                    print(get_time_string() + "%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

                if gen_iterations % save_image_frequency == 0:
                    self.save_imgs(epoch, gen_iterations, output_dir)

            self.generator.save_weights(get_checkpoint_file_name_for_epoch(generator_checkpoint_file, epoch))
            self.discriminator.save_weights(get_checkpoint_file_name_for_epoch(discriminator_checkpoint_file, epoch))

            file_name_prefix = 'validation-epoch-' + str(epoch) + '-'
            self.generate_test_sketches(YEARBOOK_TEST_PHOTOS_SAMPLE_PATH, output_dir,
                                        file_name_prefix=file_name_prefix)

        self.generator.save_weights(generator_checkpoint_file, True)
        self.discriminator.save_weights(discriminator_checkpoint_file, True)

    def save_imgs(self, epoch, gen_iterations, output_dir):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(output_dir + '/generated_image-epoch-' + str(epoch) + '-gen_iterations-' + str(gen_iterations)
                    + '.png')
        plt.close()

    def generate_test_sketches(self, test_photos_dir, output_dir,
                               file_name_prefix=''):
        photos = glob.glob(os.path.join(test_photos_dir, '*.png'))
        start = 0
        for X_test, Y_test in chunks_test(photos, BATCH_SIZE):
            X_test = (X_test.astype(np.float32) - 127.5) / 127.5

            generated_images = self.generator.predict(X_test)
            # image = combine_images(generated_images)
            # images_names = glob.glob(os.path.join('test', '*.png'))

            for i in range(len(X_test)):
                image = generated_images[i]
                image = image * 127.5 + 127.5
                image = np.swapaxes(image, 0, 2)
                image_name = photos[i + start].split('/')[-1]
                imsave(output_dir + '/' + file_name_prefix + image_name, image)

            start += len(X_test)


if __name__ == '__main__':
    if is_using_gpu():
        print('Program is using GPU..')
    else:
        print('Program is using CPU..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='Number of epochs to run', dest="epochs", default=10, type=int)
    parser.add_argument('--initial_epoch', help='Initial epoch number', dest="init_epoch", default=0, type=int)
    parser.add_argument('--train', help='Train and generate', dest="train", default=1, type=int)
    parser.add_argument('--load_weights', help='load pre-trained weights for generator and discriminator',
                        dest="load_weights", default=0, type=int)
    parser.add_argument('--generator', help='generator file name', dest="generator", default='generator')
    parser.add_argument('--discriminator', help='discriminator file name', dest="discriminator",
                        default='discriminator')
    parser.add_argument('--output_dir', help='Output directory to store intermediate images and final result',
                        dest="output_dir", default='output')

    parser.add_argument('--train_photos', help='training photos directory', dest="train_photos",
                        default='../data/yearbook_train_photos')
    parser.add_argument('--train_sketches', help='training sketches directory', dest="train_sketches",
                        default='../data/yearbook_train_sketches')
    parser.add_argument('--test_photos', help='test photos directory', dest="test_photos",
                        default='../data/yearbook_test_photos')

    args = parser.parse_args()
    print(get_time_string() + ' Args provided: ' + str(args))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    wgan = WGAN(args.load_weights, args.generator, args.discriminator)
    if args.train == 1:
        wgan.train(num_epochs=args.epochs, initial_epoch=args.init_epoch, train_photos_dir=args.train_photos,
                   train_sketches_dir=args.train_sketches, output_dir=args.output_dir,
                   generator_checkpoint_file=args.generator,
                   discriminator_checkpoint_file=args.discriminator)
    wgan.generate_test_sketches(args.test_photos, args.output_dir)
