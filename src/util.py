from conditional_gan import *
from theano import function, config, shared, tensor
import numpy as np
import time


# Returns formatted current time as string
def get_time_string():
    return time.strftime('%c') + ' '


def chunks(l, m, n):
    """Yield successive n-sized chunks from l and m."""
    for i in range(0, len(l), n):
        yield get_data_from_files(l[i: i + n], m[i: i + n])


def chunks_test(l, n):
    """Yield successive n-sized chunks from l and m."""
    for i in range(0, len(l), n):
        yield get_data_from_files(l[i: i + n])


def is_using_gpu():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = np.random.RandomState(22)
    x = shared(np.asarray(rng.rand(vlen), config.floatX))
    f = function([], tensor.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if np.any([isinstance(x.op, tensor.Elemwise) and
                       ('Gpu' not in type(x.op).__name__)
               for x in f.maker.fgraph.toposort()]):
        return False
    else:
        return True


def get_data_from_files(photo_file_names, img_rows, img_cols, sketch_file_names=None):
    data_X = np.zeros((len(photo_file_names), 3, img_cols, img_rows))
    data_Y = []

    for i in range(0, len(photo_file_names)):
        file_name = photo_file_names[i]
        data_X[i, :, :, :] = np.swapaxes(imresize(imread(file_name, mode='RGB'), (img_rows, img_cols)), 0, 2)

    if sketch_file_names:
        data_Y = np.zeros((len(sketch_file_names), 3, img_cols, img_rows))
        for i in range(0, len(sketch_file_names)):
            file_name = sketch_file_names[i]
            data_Y[i, :, :, :] = np.swapaxes(imresize(imread(file_name, mode='RGB'), (img_rows, img_cols)), 0, 2)

    return data_X, data_Y
