import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import black_box as bb


def train():
    boston = load_boston()
    df = pd.DataFrame(data=boston.data, columns=boston.feature_names)

    x = df.copy()
    y = boston.target
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_scaled[:5]

    model = keras.Sequential()
    model.add(layers.Dense(units=16, activation='relu', input_shape=(x_train_scaled.shape[1], )))
    model.add(layers.Dense(units=16, activation='relu'))
    model.add(layers.Dense(units=1))

    # model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    early_stop = keras.callbacks.EarlyStopping(patience=20)
    hist = model.fit(x_train_scaled, y_train, epochs=1000, callbacks=[early_stop], validation_split=.2)
    model.save('hello.h5')


def object_f(*params):
    model = load_model('hello.h5')
    inputs = np.array(*params).reshape(-1, 13)
    pred = model.predict(inputs)
    print(pred[0][0])
    return pred[0][0]


def main():
    train()
    best_params = bb.search_min(f=object_f, domain = [
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                                [0., 1.],
                            ],
                            budget = 30,
                            batch = 1,
                            resfile = 'output.csv')


if __name__ == '__main__':
    main()
