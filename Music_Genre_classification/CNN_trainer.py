import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

dataset_path = "Generative_AI/data.json"


def load_data(dataset_path):

    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    input_features = np.array(data["mfcc"])
    output_targets = np.array(data["labels"])  # like annotated labels(targets)

    return input_features, output_targets


def dataset_split(test_size, validation_size):

    input_features, output_targets = load_data(dataset_path=dataset_path)
    '''
    (100%)
    |
    |___ Testing (25%)
    |
    |___ Training (65%)
    |       |
    |       |___ Validation (20%)
    '''
    feature_train, feature_test, label_train, label_test = train_test_split(
        input_features, output_targets, test_size=test_size)

    feature_train, feature_validation, label_train, label_validation = train_test_split(
        feature_train, label_train, test_size=validation_size)

    '''output is of MFCC is ℝ³ (9986, 130, 13)
       CNN feature input needs one more dimension: color channel (9986, 130, 13, 1).
    '''
    # turn features ℝ³ → ℝ⁴

    feature_train = feature_train[..., np.newaxis]
    feature_validation = feature_validation[..., np.newaxis]
    feature_test = feature_test[..., np.newaxis]

    return feature_train, feature_validation, feature_test, label_train, label_validation, label_test


def cnn_model(input_shape):

    model = Sequential()

    # ------------------------------------
    # Conv Block 1: 32 Filters, MaxPool.
    # ------------------------------------
    model.add(Conv2D(filters=32, kernel_size=3, padding='same',
              activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3,
              padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ------------------------------------
    # Conv Block 2: 64 Filters, MaxPool.
    # ------------------------------------
    model.add(Conv2D(filters=32, kernel_size=3,
              padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3,
              padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ------------------------------------
    # Conv Block 3: 64 Filters, MaxPool.
    # ------------------------------------
    model.add(Conv2D(filters=32, kernel_size=2,
              padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=2,
              padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ------------------------------------
    # Flatten the convolutional features.
    # ------------------------------------
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))

    # output of the layer is Softmax Probabilities tensor
    model.add(Dense(10, activation='softmax'))

    return model


def predict(model, y_hat, x):

    y_hat = y_hat[np.newaxis, ...]

    predictions = model.predict(y_hat)

    predicted_index = np.argmax(predictions, axis=1)

    print("expectd : {} \nPredicted : {}".format(x, predicted_index))


if __name__ == "__main__":
    # create training , validation and test sets
    feature_train, feature_validation, feature_test, label_train, label_validation, label_test = dataset_split(
        test_size=0.25, validation_size=0.20)

    # build CNN model
    model = cnn_model(input_shape=(
        feature_train.shape[1], feature_train.shape[2], feature_train.shape[3]))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # compile the model
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # train the model
    model.fit(feature_train, label_train, validation_data=(feature_validation, label_validation), epochs=30,
              batch_size=32)

    # evaluate the CNN on test set
    test_error, test_accuracy = model.evaluate(
        feature_test, label_test, verbose=1)
    print("Accuracy is : {}".format(test_accuracy))

    # make presictions on sample
    predicted = feature_test[100]
    ground_truth = label_test[100]

    predict(model=model, y_hat=predicted, x=ground_truth)
