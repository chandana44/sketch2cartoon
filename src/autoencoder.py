
import os

os.environ['THEANO_FLAGS'] = "device=cuda0"

from util import *
import util

import argparse
import glob
import math

import numpy as np
from keras import backend as K
from keras.layers import Input, merge, Convolution2D, MaxPooling2D
from keras.layers.convolutional import Deconvolution2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from scipy.misc import imsave

K.set_image_dim_ordering('th')

IN_CH = 3
OUT_CH = 3
LAMBDA = 100
NF = 64  # number of filter

YEARBOOK_TEST_PHOTOS_SAMPLE_PATH = '../data/yearbook_test_photos_sample'


def generator_model(img_rows, img_cols):
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

    d1 = Deconvolution2D(512, 5, 5, subsample=(2, 2), activation='relu', init='uniform', output_shape=(None, 512, 2, 2),
                         border_mode='same')(e6)
    d1 = merge([d1, e5], mode='concat', concat_axis=1)
    d1 = BatchNormalization()(d1)

    d2 = Deconvolution2D(512, 5, 5, subsample=(2, 2), activation='relu', init='uniform', output_shape=(None, 512, 4, 4),
                         border_mode='same')(d1)
    d2 = merge([d2, e4], mode='concat', concat_axis=1)
    d2 = BatchNormalization()(d2)

    d3 = Dropout(0.2)(d2)
    d3 = Deconvolution2D(512, 5, 5, subsample=(2, 2), activation='relu', init='uniform', output_shape=(None, 512, 8, 8),
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

    d6 = Deconvolution2D(3, 5, 5, subsample=(2, 2), activation='relu', init='uniform', output_shape=(None, 3, 64, 64),
                         border_mode='same')(d5)

    d6 = BatchNormalization()(d6)
    d7 = Activation('tanh')(d6)

    model = Model(input=inputs, output=d7)
    return model

# def generator_l1_loss(y_true, y_pred):
#     return K.mean(K.abs(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)


def get_checkpoint_file_name_for_epoch(checkpoint_file_name_base, epoch):
    return checkpoint_file_name_base + '-epoch-' + str(epoch)


def train(LOAD_WEIGHTS, EPOCHS, INIT_EPOCH, train_photos_dir, train_sketches_dir, output_dir,
          generator_checkpoint_file, discriminator_checkpoint_file, img_rows, img_cols, batch_size):
    photos = glob.glob(os.path.join(train_photos_dir, '*.png'))
    sketches = glob.glob(os.path.join(train_sketches_dir, '*.png'))

    # discriminator = discriminator_model()
    generator = generator_model(img_rows, img_cols)

    if LOAD_WEIGHTS == 1:
        generator.load_weights(generator_checkpoint_file)
        # discriminator.load_weights(discriminator_checkpoint_file)

    generator.summary()
    # discriminator.summary()

    # discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    generator.compile(loss='mse', optimizer="rmsprop")
    # discriminator_on_generator.compile(loss=[generator_l1_loss, discriminator_on_generator_loss], optimizer="rmsprop")
    # discriminator.trainable = True
    # discriminator.compile(loss=discriminator_loss, optimizer="rmsprop")

    for epoch in range(INIT_EPOCH, EPOCHS):
        index = 0
        print(get_time_string() + " Epoch is", epoch)
        print(get_time_string() + " Number of batches", int(len(photos) / batch_size))

        for X_train, Y_train in chunks(photos, sketches, batch_size, img_rows, img_cols):
            print(get_time_string() + ' Batch number: ' + str(index))
            X_train = (X_train.astype(np.float32) - 127.5) / 127.5
            Y_train = (Y_train.astype(np.float32) - 127.5) / 127.5

            print(get_time_string() + " Training the generator...")
            g_loss = generator.train_on_batch(X_train, Y_train)
            print(get_time_string() + " batch %d g_loss : %f" % (index, g_loss))

            if index % 20 == 0:
                generator.save_weights(generator_checkpoint_file, True)
                # discriminator.save_weights(discriminator_checkpoint_file, True)
            index += 1

        generator.save_weights(get_checkpoint_file_name_for_epoch(generator_checkpoint_file, epoch))

        file_name_prefix = 'validation-epoch-' + str(epoch) + '-'
        generate(YEARBOOK_TEST_PHOTOS_SAMPLE_PATH, output_dir, generator_checkpoint_file, discriminator_checkpoint_file,
                 img_rows,img_cols, batch_size, file_name_prefix=file_name_prefix)

    generator.save_weights(generator_checkpoint_file, True)


def generate(test_photos_dir, output_dir, generator_checkpoint_file, discriminator_checkpoint_file,
             img_rows, img_cols, batch_size, file_name_prefix=''):
    photos = glob.glob(os.path.join(test_photos_dir, '*.png'))
    start = 0
    for X_test, Y_test in chunks_test(photos, batch_size, img_rows, img_cols):
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5
        generator = generator_model(img_rows, img_cols)
        generator.compile(loss='binary_crossentropy', optimizer="SGD")
        generator.load_weights(generator_checkpoint_file)

        generated_images = generator.predict(X_test)
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

    parser.add_argument('--img_rows', help='num rows', dest="img_rows",
                        default=64, type=int)
    parser.add_argument('--img_cols', help='num cols', dest="img_cols",
                        default=64, type=int)
    parser.add_argument('--batch_size', help='batch size', dest="batch_size",
                        default=128, type=int)

    args = parser.parse_args()
    print(get_time_string() + ' Args provided: ' + str(args))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.train == 1:
        train(args.load_weights, args.epochs, args.init_epoch, args.train_photos, args.train_sketches, args.output_dir,
              args.generator, args.discriminator, args.img_rows, args.img_cols, args.batch_size)
    generate(args.test_photos, args.output_dir, args.generator, args.discriminator, args.img_rows, args.img_cols,
             args.batch_size)
