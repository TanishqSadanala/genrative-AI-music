import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

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


def plot_history(history):
    fig, axes = plt.subplots(2)
    axes[0].plot(history.history["accuracy"], label="train_accuracy")
    axes[0].plot(history.history["val_accuracy"], label="testing_accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("epochs")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Accuracy graph")
    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="testing_loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("epochs")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Loss graph")
    plt.show()


def Lstm_model(input_shape):
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

if __name__ == "__main__":
    input_features, output_targets = load_data(dataset_path)
    print(input_features.shape)
    
    feature_train, feature_validation, feature_test, label_train, label_validation, label_test = dataset_split(
        test_size=0.25, validation_size=0.20)
    
    model = Lstm_model(input_shape=(feature_train.shape[1], feature_train.shape[2]))

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(feature_train, label_train, validation_data=(feature_validation, label_validation), epochs=30,
              batch_size=32)

    plot_history(history)
    
    test_error, test_accuracy = model.evaluate(
        feature_test, label_test, verbose=1)
    print("Accuracy is : {}".format(test_accuracy))

