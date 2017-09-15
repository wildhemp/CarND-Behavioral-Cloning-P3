import numpy as np
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
from collections import Counter

from keras.regularizers import l2, activity_l2
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Activation, Cropping2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 3, "Epochs to run the model for")
flags.DEFINE_integer('batch_size', 128, "Batch size")
flags.DEFINE_boolean('visualize', False, "Whether to visualize the data or not")
flags.DEFINE_boolean('train', True, "Whether to train the model or not")


def prepare_data(dataset, path, augment=False):
    lines = []
    original = []
    for _, line in enumerate(dataset):
        if float(line[6]) == .0:
            continue

        angle = float(line[3])
        image_path = path + 'IMG/' + line[0].decode('utf-8').split('/')[-1]

        # store the original angles, so that we can visualize them and compare to the augmented dataset
        original.append((image_path, angle))

        if not augment:
            lines.append((image_path, angle))
            continue

        if abs(angle) == .0 and np.random.uniform() < 2./3.:
            continue

        lines.append((image_path, angle))

        include_both = abs(angle) > .0

        correction_angle = .09

        left_correct = min(1., angle + correction_angle)
        right_correct = max(-1., angle - correction_angle)

        choice = np.random.uniform()
        if include_both or choice < 1./3.:
            image_path = path + 'IMG/' + line[1].decode('utf-8').split('/')[-1]
            lines.append((image_path, left_correct))
        if include_both or choice >= 2./3.:
            image_path = path + 'IMG/' + line[2].decode('utf-8').split('/')[-1]
            lines.append((image_path, right_correct))

    data = np.array(lines, dtype=([('path', '|S200'), ('angle', float)]))

    return data, np.array(original, dtype=([('path', '|S200'), ('angle', float)]))


def load_data(path='data/'):
    log = np.genfromtxt(path + 'driving_log.csv', delimiter=',', dtype=None, skip_header=1)

    size = len(log)
    train = log[:int(size * .8 - .5)]
    valid = log[int(size * .8 + .5):]

    train, train_orig = prepare_data(train, path, augment=True)
    valid, valid_orig = prepare_data(valid, path, augment=False)

    return train, valid, np.concatenate((train_orig, valid_orig))


def dataset_info(dataset, title, column='angle', steps=0.02):
    angles = dataset[column]
    max = np.max(angles)
    min = np.min(angles)
    print('{}: #: {} , Min: {:.4f}, Max: {:.4f}'.format(title, dataset.shape, min, max))
    plt.figure().canvas.set_window_title(title)
    plt.hist([angle for angle in angles if abs(angle) >= .0], bins=np.arange(min, max, steps))


def visualize_data(train, valid, original):
    dataset_info(train, 'Training data')
    dataset_info(valid, 'Validation data')
    dataset_info(original, 'Original data')

    index = 6291 #np.random.randint(0, len(original))
    print(index)
    center_path = original[index]['path'].decode("utf-8")
    path = center_path.split('/')
    right_path = os.path.join(path[0], path[1], 'right' + path[2][6:])
    left_path = os.path.join(path[0], path[1], 'left' + path[2][6:])

    image = cv2.imread(center_path)
    cv2.imshow('Center, angle: {:.4f}'.format(original[index]['angle']), image)

    image, angle = augment(image, original[index]['angle'])
    cv2.imshow('Center (agumented), angle: {:.4f}'.format(angle), image)

    image = cv2.imread(right_path)
    cv2.imshow('Right, angle: {:.4f}'.format(original[index]['angle']), image)

    image = cv2.imread(left_path)
    cv2.imshow('Left, angle: {:.4f}'.format(original[index]['angle']), image)

    plt.show()


def augment(image, angle):
    oh, ow = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = np.array(image, dtype = np.float32)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (image[:, :, 0] + value) > 255
    if value <= 0:
        mask = (image[:, :, 0] + value) < 0

    image[:, :, 0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h, w = image.shape[0:2]
    mid = np.random.randint(0, w)
    factor = np.random.uniform(0.6, 0.8)
    if np.random.rand() > .5:
        image[:, 0:mid, 0] *= factor
    else:
        image[:, mid:w, 0] *= factor

    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)

    if angle > .25:
        tx = np.random.uniform(0., 35.)
    elif angle < -.25:
        tx = np.random.uniform(-35., 0.)
    else:
        tx = np.random.uniform(-35., 35.)

    M = np.float32(([1, 0, tx], [0, 1, 0]))

    return cv2.warpAffine(image, M, (ow, oh)), angle + tx * .009


def generator(data, bach_size=128):
    num_data = len(data)
    while True:
        data = shuffle(data)
        for offset in range(0, num_data, bach_size):
            batch = data[offset:offset + bach_size]

            images = []
            angles = []
            for current in batch:
                image = cv2.cvtColor(cv2.imread(current['path'].decode("utf-8")), cv2.COLOR_BGR2RGB)
                angle = float(current['angle'])
                if np.random.uniform() < .5:
                    image = np.fliplr(image)
                    angle = -angle

                if np.random.uniform() < .5:
                    image, angle = augment(image, angle)

                images.append(image), angles.append(angle)

            X = np.array(images)
            y = np.array(angles)

            yield shuffle(X, y)


def build_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((80, 30), (35, 35)), input_shape=(160, 320, 3)))
    # normalize image
    model.add(Lambda(lambda x: (x / 127.5) - 1.))

    model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(.001)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(.001)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode='same', W_regularizer=l2(.001)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode='same', W_regularizer=l2(.001)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(1000, W_regularizer=l2(.001)))
    model.add(Activation('relu'))

    model.add(Dense(250, W_regularizer=l2(.001)))
    model.add(Activation('relu'))

    model.add(Dense(50, W_regularizer=l2(.001)))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam())

    return model


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_1, valid_1, original_1 = load_data()
    train_2, valid_2, original_2 = load_data(path='data_track_2/')
    train, valid, original = np.concatenate((train_1, train_2)), \
                             np.concatenate((valid_1, valid_2)),\
                             np.concatenate((original_1, original_2))

    train_generator = generator(train)
    valid_generator = generator(valid)

    model = build_model()

    if FLAGS.visualize:
        model.summary()
        visualize_data(train, valid, original)

    if FLAGS.train:
        model.fit_generator(train_generator, samples_per_epoch=len(train), nb_epoch=FLAGS.epochs,
                            validation_data=valid_generator, nb_val_samples=len(valid))

        model.save('model.h5')


if __name__ == '__main__':
    tf.app.run()
