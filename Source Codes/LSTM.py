import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# for file operations
import os
import keras.backend as K
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.utils import plot_model
import gc

# natural sorting using regular expressions
import re
_nsre = re.compile('([0-9]+)')


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


# using seed for reproducibility
np.random.seed(2016)

# using tensorflow backend
K.set_image_dim_ordering('tf')

# specify the path to KTH data folder
trg_data_root = "D:\Documents\KTH\\"


# load training or validation data
# with 25 persons in the dataset, start_index and finish_index has to be in the range [1..25]
def load_data_for_persons(start_index, finish_index):
    # these strings are needed for creating subfolder names
    class_labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]  # 6 labels
    frame_path = "\\frames\\"
    frame_set_prefix = "person"  # 2 digit person ID [01..25] follows
    rec_prefix = "d"  # seq ID [1..4] follows
    rec_count = 4
    seg_prefix = "seg"  # seq ID [1..4] follows
    seg_count = 4
    frames_per_clip = 25
    data_array = []
    classes_array = []

    # a couple of loops to generate the data
    for i in range(0, len(class_labels)):
        # class
        class_folder = trg_data_root + class_labels[i] + frame_path

        for j in range(start_index, finish_index+1):
            # person
            print("\nLoading data of ", class_labels[i], " of person ", j)
            if j < 10:
                person_folder = class_folder + frame_set_prefix + "0" + str(j) + "_" + class_labels[i] + "_"
            else:
                person_folder = class_folder + frame_set_prefix + str(j) + "_" + class_labels[i] + "_"

            for k in range(1, rec_count+1):
                # recording
                rec_folder = person_folder + rec_prefix + str(k) + "\\"
                for m in range(1, seg_count+1):
                    # segment
                    seg_folder = rec_folder + seg_prefix + str(m) + "\\"

                    # get the list of files
                    file_list = [f for f in os.listdir(seg_folder)]
                    example_size = len(file_list)

                    # for larger segments, we can change the starting point to augment the data
                    clip_start_index = 0
                    if example_size > frames_per_clip:
                        # sample the frames from the center
                        clip_start_index = example_size/2 - frames_per_clip/2
                        example_size = frames_per_clip

                    # need natural sort before loading data
                    file_list.sort(key=natural_sort_key)

                    # create a list for each segment
                    for n in range(int(clip_start_index), int(example_size+clip_start_index)):
                        file_path = seg_folder + file_list[i]
                        data = np.asarray(cv2.imread(file_path), dtype='uint8')

                    # preprocessing
                    current_seg = np.asarray(data)
                    current_seg = current_seg.astype('float32')

                    data_array.append(current_seg)
                    dataarray = np.delete(data_array, [1, 2], 3)
                    classes_array.append(i)

    # create one-hot vectors from output values
    classes_one_hot = np.zeros((len(classes_array), len(class_labels)))
    classes_one_hot[np.arange(len(classes_array)), classes_array] = 1

    data_array = np.array(dataarray)
    # data_array.reshape((1, 2400, 120, 160, 3))
    # done
    return data_array, classes_one_hot


# what you need to know about data, to build the model
img_rows = 120
img_cols = 160
nb_classes = 6
class_labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

# build network model
print("Building model")
time_model_start = time.time()
# define our time-distributed setup
model = Sequential()

# three convolutional layers
model.add(Conv2D(4, (5, 5), strides=(2, 2), padding='valid', input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))


model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='valid'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='valid'))
model.add(Activation('relu'))


# flatten and prepare to go for recurrent learning
model.add(TimeDistributed(Flatten()))

# a single dense layer
model.add(TimeDistributed(Dense(80)))
model.add(BatchNormalization())  # required for ensuring that the network learns
model.add(Activation('relu'))


# the LSTM layer
model.add(LSTM(80, activation='tanh'))

# let's try some dropping out here
model.add(Dropout(0.1))

# fully connected layers to finish off
model.add(Dense(80, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# training parameters
batchsize = 64  # increase if your system can cope with more data
nb_epochs = 30
model.summary()
plot_model(model, "D:\Documents\KTH\Model_LSTM.png")
time_model_end = time.time()
print("Loading data")
time_load_start = time.time()
start_index_train = 1
finish_index_train = 25
# load training data
X_train, y_train = load_data_for_persons(start_index_train, finish_index_train)
time_load_end = time.time()
# if you can't fit all data in memory, load a few users at a time and
# use multiple epochs. I don't recommend using one user at a time, since
# it prevents good shuffling.

# perform training
print("Training")
time_train_start = time.time()
model.fit(np.array(X_train), y_train, batch_size=batchsize, epochs=nb_epochs, shuffle=True, verbose=1)

# clean up the memory
X_train = None
y_train = None
gc.collect()
time_train_end = time.time()
print("Testing")
time_test_start = time.time()
# load test data
start_index_test = 21
finish_index_test = 25
X_test, y_test = load_data_for_persons(start_index_test, finish_index_test)


preds = model.predict(np.array(X_test))

confusion_matrix = np.zeros(shape=(y_test.shape[1], y_test.shape[1]))
accurate_count = 0.0
for i in range(0, len(preds)):
    # updating confusion matrix
    confusion_matrix[np.argmax(preds[i])][np.argmax(np.array(y_test[i]))] += 1

    # Axes of confusion matrix
    print('Predicted: ', np.argmax(preds[i]), ', actual: ', np.argmax(np.array(y_test[i])))

    # calculating overall accuracy
    if np.argmax(preds[i]) == np.argmax(np.array(y_test[i])):
        accurate_count += 1
fig = plt.figure(figsize=(12, 10), dpi=100)
plt.imshow(confusion_matrix, interpolation='nearest', cmap='Pastel1')
plt.title("HAR using CNN-LSTM")
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
for o in range(6):
    for p in range(6):
        plt.text(p, o, confusion_matrix[o][p].astype(int), horizontalalignment="center", verticalalignment="center")
plt.savefig("D:\Documents\KTH\Confusion_Matrix_LSTM.png")
time_test_end = time.time()
print("Number of epochs trained for: ", nb_epochs)
print('Total no. of testing samples used:', y_test.shape[0])
print('Validation accuracy: ', 100*accurate_count/len(preds), ' %')
print('Confusion matrix:')
print(class_labels)
print(confusion_matrix)
print("Time taken to build the model: ", time_model_end - time_model_start)
print("Time taken to load the data: ", time_load_end - time_load_start)
print("Time taken to train the samples: ", time_train_end - time_train_start)
print("Time taken to test the samples: ", time_test_end - time_test_start)
# save the model
jsonstring = model.to_json()
with open("D:\Documents\KTH\KTH_LSTM.json", 'w') as f:
    f.write(jsonstring)
model.save_weights("D:\Documents\KTH\KTH_LSTM.h5", overwrite=True)
