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
import util
import sys

import numpy as np

K.set_image_dim_ordering('th')

img_rows = util.img_rows
img_cols = util.img_cols
IN_CH = util.IN_CH
OUT_CH = util.OUT_CH
LAMBDA = util.LAMBDA
NF = util.NF  # number of filter
BATCH_SIZE = util.BATCH_SIZE

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

        self.combined = generator_containing_discriminator(self.generator, self.discriminator)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def generator_containing_discriminator(self, generator, discriminator):
        inputs = Input((IN_CH, img_cols, img_rows))
        x_generator = generator(inputs)

        merged = merge([inputs, x_generator], mode='concat', concat_axis=1)
        discriminator.trainable = False
        x_discriminator = discriminator(merged)

        model = Model(input=inputs, output=[x_generator, x_discriminator])

        return model

    def build_generator(self):

        global BATCH_SIZE
        # imgs: input: 256x256xch
        # U-Net structure, must change to relu
        inputs = Input((IN_CH, img_cols, img_rows))

        e1 = BatchNormalization()(inputs)
        e1 = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', init='uniform', border_mode='same')(e1)
        e1 = BatchNormalization()(e1)
        e2 = Convolution2D(128, 4, 4, subsample=(2, 2), activation='relu', init='uniform', border_mode='same')(e1)
        e2 = BatchNormalization()(e2)

        e3 = Convolution2D(256, 4, 4, subsample=(2, 2), activation='relu', init='uniform', border_mode='same')(e2)
        e3 = BatchNormalization()(e3)
        e4 = Convolution2D(512, 4, 4, subsample=(2, 2), activation='relu', init='uniform', border_mode='same')(e3)
        e4 = BatchNormalization()(e4)

        e5 = Convolution2D(512, 4, 4, subsample=(2, 2), activation='relu', init='uniform', border_mode='same')(e4)
        e5 = BatchNormalization()(e5)
        e6 = Convolution2D(512, 4, 4, subsample=(2, 2), activation='relu', init='uniform', border_mode='same')(e5)
        e6 = BatchNormalization()(e6)

        d1 = Deconvolution2D(512, 5, 5, subsample=(2, 2), activation='relu', init='uniform',
                             output_shape=(None, 512, 2, 2),
                             border_mode='same')(e6)
        d1 = merge([d1, e5], mode='concat', concat_axis=1)
        d1 = BatchNormalization()(d1)

        d2 = Deconvolution2D(512, 5, 5, subsample=(2, 2), activation='relu', init='uniform',
                             output_shape=(None, 512, 4, 4),
                             border_mode='same')(d1)
        d2 = merge([d2, e4], mode='concat', concat_axis=1)
        d2 = BatchNormalization()(d2)

        d3 = Dropout(0.2)(d2)
        d3 = Deconvolution2D(512, 5, 5, subsample=(2, 2), activation='relu', init='uniform',
                             output_shape=(None, 512, 8, 8),
                             border_mode='same')(d3)
        d3 = merge([d3, e3], mode='concat', concat_axis=1)
        d3 = BatchNormalization()(d3)

        d4 = Dropout(0.2)(d3)
        d4 = Deconvolution2D(512, 5, 5, subsample=(2, 2), activation='relu', init='uniform',
                             output_shape=(None, 512, 16, 16), border_mode='same')(d4)
        d4 = merge([d4, e2], mode='concat', concat_axis=1)
        d4 = BatchNormalization()(d4)

        d5 = Dropout(0.2)(d4)
        d5 = Deconvolution2D(256, 5, 5, subsample=(2, 2), activation='relu', init='uniform',
                             output_shape=(None, 256, 32, 32), border_mode='same')(d5)
        d5 = merge([d5, e1], mode='concat', concat_axis=1)
        d5 = BatchNormalization()(d5)

        d6 = Deconvolution2D(3, 5, 5, subsample=(2, 2), activation='relu', init='uniform',
                             output_shape=(None, 3, 64, 64),
                             border_mode='same')(d5)

        d6 = BatchNormalization()(d6)
        d7 = Activation('tanh')(d6)

        model = Model(input=inputs, output=d7)
        return model

    def build_discriminator(self):

        """ return a (b, 1) logits"""
        model = Sequential()
        model.add(Convolution2D(64, 4, 4, border_mode='same', input_shape=(IN_CH, img_cols, img_rows)))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 4, 4, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(512, 4, 4, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Convolution2D(1, 4, 4, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))

        model.add(Activation('sigmoid'))
        return model

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

                image_batch = Y_train
                print(get_time_string() + " Predicting...")
                generated_images = self.generator.predict(X_train)
                if index % 50 == 0:
                    image = combine_images(generated_images)
                    image = image * 127.5 + 127.5
                    image = np.swapaxes(image, 0, 2)
                    imsave(output_dir + "/epoch-" + str(epoch) + "_batch-" + str(index) + ".png", image)
                    # Image.fromarray(image.astype(np.uint8)).save(str(epoch)+"_"+str(index)+".png")

                gen_iterations += 1

                if gen_iterations <= 225 or gen_iterations % 2500 == 0:
                    self.n_critic = 1
                else:
                    self.n_critic = 5

                # ---------------------
                #  Train Discriminator
                # ---------------------

                print(get_time_string() + " Training the discriminator...")

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(image_batch, np.ones((len(X_train), 1, 64, 64)))
                d_loss_fake = self.discriminator.train_on_batch(generated_images, -np.ones((len(X_train), 1, 64, 64)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # pred_temp = discriminator.predict(X)
                # print(np.shape(pred_temp))
                print(get_time_string() + " batch %d d_loss : %f" % (index, d_loss))

                # Clip discriminator weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

                # ---------------------
                #  Train Generator
                # ---------------------
                if gen_iterations % self.n_critic == 0:
                    # Train the generator
                    g_loss = self.combined.train_on_batch.train_on_batch(X_train, [image_batch, np.ones((len(X_train), 1, 64, 64))])

                    # Plot the progress
                    print(get_time_string() + "%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

                if gen_iterations % save_image_frequency == 0:
                    self.save_imgs(epoch, gen_iterations, output_dir)

                if index % 20 == 0:
                    self.generator.save_weights(generator_checkpoint_file, True)
                    self.discriminator.save_weights(discriminator_checkpoint_file, True)
                index += 1

            self.generator.save_weights(get_checkpoint_file_name_for_epoch(generator_checkpoint_file, epoch))
            self.discriminator.save_weights(get_checkpoint_file_name_for_epoch(discriminator_checkpoint_file, epoch))

            file_name_prefix = 'validation-epoch-' + str(epoch) + '-'
            generate(YEARBOOK_TEST_PHOTOS_SAMPLE_PATH, output_dir, generator_checkpoint_file,
                     discriminator_checkpoint_file,
                     file_name_prefix=file_name_prefix)

        self.generator.save_weights(generator_checkpoint_file, True)
        self.discriminator.save_weights(discriminator_checkpoint_file, True)  # Select a random half batch of images

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
