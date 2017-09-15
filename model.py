import numpy as np
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

from keras.regularizers import l2
from keras.layers import Dense, Flatten, Lambda, Conv2D, Activation, Cropping2D
from keras.models import Sequential
from keras.optimizers import Adam

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 3, "Epochs to run the model for")
flags.DEFINE_integer('batch_size', 128, "Batch size")
flags.DEFINE_boolean('visualize', False, "Whether to visualize the data or not")
flags.DEFINE_boolean('train', True, "Whether to train the model or not")


def prepare_data(dataset, path, augment=False):
    '''
    Prepares the dataset. This constitutes of
        * filtering out 2/3 of the 0 angles
        * filtering out everything where speed is 0
        * figuring out the path for the image files
        * adding either all (when angle is > 0) or 1/3 of left and 1/3 of right camera images. The
          correction angle applies is chosen by doing a back of the envelope calculation of how
          much we'd need to add to the angle for the car to get back in the center image position
          in ~2 secs assuming 30mph/50kph speed.

    :param dataset: the dataset as it is loaded from the csv file
    :param path: the path the images/csv are stored
    :param augment: whether the dataset should be augmented with left/right images and filtered.
        The validation dataset is not augmented.
    :return: a tuple of (path, angle) for the (maybe) augmented/filtered data and a tuple for the
        original data (to be used for e.g. visualization).
    '''
    lines = []
    original = []
    for _, line in enumerate(dataset):
        # leave out everything which has 0 speed.
        if float(line[6]) == .0:
            continue

        angle = float(line[3])
        image_path = path + 'IMG/' + line[0].decode('utf-8').split('/')[-1]

        # Store the original angles, so that we can visualize them and compare to the augmented
        # dataset.
        original.append((image_path, angle))

        if not augment:
            lines.append((image_path, angle))
            continue

        # There are too much data with 0 angle, so leave out 2/3 to not bias our model too much.
        if abs(angle) == .0 and np.random.uniform() < 2./3.:
            continue

        lines.append((image_path, angle))

        include_both = abs(angle) > .0

        # From back of the envelope calculation much we'd need to add to the angle for the car to
        # get back in the center image position in ~2 secs assuming 30mph/50kph speed.
        # The calculation actually says 0.075, but added a bit more so that it handles corners
        # better.
        correction_angle = .09

        left_correct = min(1., angle + correction_angle)
        right_correct = max(-1., angle - correction_angle)

        # If the angle is > 0, we include both left and right camera image to have a more balanced
        # dataset.
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
    '''
    Loads the data from the given path, split it to train and validation sets and prepares them
    using #prepare_data(...). The validation set is simply the last 20% of the dataset. Since the
    data is images and angles of driving this ensures that the validation dataset has the same
    distribution as the whole dataset and the test track.

    :param path: the path to load the data from. Defaults to 'data/'
    :return: numpy arrays for the training, validation and the original dataset.
    '''
    log = np.genfromtxt(path + 'driving_log.csv', delimiter=',', dtype=None, skip_header=1)

    # Split the data to training/validation 80/20.
    size = len(log)
    train = log[:int(size * .75 - .5)]
    valid = log[int(size * .75 + .5):]

    train, train_orig = prepare_data(train, path, augment=True)
    valid, valid_orig = prepare_data(valid, path, augment=False)

    return train, valid, np.concatenate((train_orig, valid_orig))


def dataset_info(dataset, title, column='angle', steps=0.02):
    '''
    Prints information and plots bar chart of the dataset.

    :param dataset: the dataset to have the information about
    :param title: the title of the chart
    :param column: the column in the dataset to print the info about
    :param steps: the steps for showing the bar chart
    '''
    angles = dataset[column]
    max = np.max(angles)
    min = np.min(angles)
    print('{}: #: {} , Min: {:.4f}, Max: {:.4f}'.format(title, dataset.shape, min, max))

    plt.figure().canvas.set_window_title(title)
    plt.hist([angle for angle in angles if abs(angle) >= .0], bins=np.arange(min, max, steps))


def visualize_data(train, valid, original):
    '''
    Visualizes the data, showing bar charts and sample images.

    :param train: the train dataset
    :param valid: the validation dataset
    :param original: the original dataset
    '''
    dataset_info(train, 'Training data')
    dataset_info(valid, 'Validation data')
    dataset_info(original, 'Original data')

    index = 6291 #np.random.randint(0, len(original))

    center_path = original[index]['path'].decode("utf-8")
    path = center_path.split('/')
    right_path = os.path.join(path[0], path[1], 'right' + path[2][6:])
    left_path = os.path.join(path[0], path[1], 'left' + path[2][6:])

    image = cv2.imread(center_path)
    cv2.imshow('Center, angle: {:.4f}'.format(original[index]['angle']), image)

    image, angle = augment(image, original[index]['angle'])
    cv2.imshow('Center (augmented), angle: {:.4f}'.format(angle), image)

    image = cv2.imread(right_path)
    cv2.imshow('Right, angle: {:.4f}'.format(original[index]['angle']), image)

    image = cv2.imread(left_path)
    cv2.imshow('Left, angle: {:.4f}'.format(original[index]['angle']), image)


def augment(image, angle):
    '''
    Augments the data by applying brightness, darkening and horizontal shifting to the image. The
    code was partially taken from
    https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project/blob/747f21f02219e1743367fdc43aeb37993cafe08a/model.py#L89

    :param image: the image to augment
    :param angle: the angle belonging to the image
    :return: the new image and angle
    '''
    oh, ow = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = np.array(image, dtype = np.float32)
    # Random brightness - the mask bit keeps values from going beyond [0, 255]
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (image[:, :, 0] + value) > 255
    if value <= 0:
        mask = (image[:, :, 0] + value) < 0

    image[:, :, 0] += np.where(mask, 0, value)
    # Random shadow - full height, random left/right side, random darkening
    h, w = image.shape[0:2]
    mid = np.random.randint(0, w)
    factor = np.random.uniform(0.6, 0.8)
    if np.random.rand() > .5:
        image[:, 0:mid, 0] *= factor
    else:
        image[:, mid:w, 0] *= factor

    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)

    # Random horizontal translation - when the angle is high, we make sure to translate, so that
    # it's even higher, i.e. the car is turning even more. The assumption is, this will make the
    # car not run out of the road on sharp turns. Otherwise we just randomly select a translation
    # to either direction.
    if angle > .25:
        tx = np.random.uniform(0., 25.)
    elif angle < -.25:
        tx = np.random.uniform(-25., 0.)
    else:
        tx = np.random.uniform(-25., 25.)

    M = np.float32(([1, 0, tx], [0, 1, 0]))
    image = cv2.warpAffine(image, M, (ow, oh))
    angle += tx * .009

    horizon = 2 * oh / 5
    v_shift = np.random.randint(int(-oh / 8), int(oh / 8))
    pts1 = np.float32([[0, horizon], [ow, horizon], [0, oh], [ow, oh]])
    pts2 = np.float32([[0, horizon + v_shift], [ow, horizon + v_shift], [0, oh], [ow, oh]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, M, (ow, oh), borderMode=cv2.BORDER_REPLICATE)

    return image, angle


def generator(data, bach_size=128):
    '''
    Generates the inputs for training the model.

    :param data: the dataset to use (e.g. can be validation or training)
    :param bach_size: the batch size.
    :return: features, angles tuple
    '''
    num_data = len(data)
    while True:
        # Shuffle the data every time we start over, so that the order doesn't skew our model.
        data = shuffle(data)
        for offset in range(0, num_data, bach_size):
            batch = data[offset:offset + bach_size]

            images = []
            angles = []
            for current in batch:
                # Read and conver the image from BGR to RGB. Interestingly, the model works quite
                # well on RGB data when trained on BGR, but run of the road at some points...
                image = cv2.cvtColor(cv2.imread(current['path'].decode("utf-8")), cv2.COLOR_BGR2RGB)
                angle = float(current['angle'])
                # Flip the images half the time, to help the model generalize better.
                if np.random.uniform() < .5:
                    image = np.fliplr(image)
                    angle = -angle

                # Apply random augmentation half the time, again, helping the model generalize.
                if np.random.uniform() < .5:
                    image, angle = augment(image, angle)

                images.append(image), angles.append(angle)

            X = np.array(images)
            y = np.array(angles)

            yield shuffle(X, y)


def build_model():
    '''
    Builds the model.

    :return: the model
    '''
    model = Sequential()
    model.add(Cropping2D(cropping=((80, 30), (25, 25)), input_shape=(160, 320, 3)))
    # Normalize image
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


def visualize_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model MSE loss')
    plt.ylabel('MSE loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')


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
        history = model.fit_generator(
            train_generator, samples_per_epoch=len(train), nb_epoch=FLAGS.epochs,
            validation_data=valid_generator, nb_val_samples=len(valid))
        visualize_loss(history)

        model.save('model.h5')

    plt.show()

if __name__ == '__main__':
    tf.app.run()
