import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utilities import Hyperparameters

SCRIPT_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_REL_PATH = 'data/'
RESULTS_DIR_REL_PATH = 'results/baseline-' + datetime.now().isoformat(' ', 'seconds') + '/'
DATA_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, DATA_DIR_REL_PATH)
RESULTS_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, RESULTS_DIR_REL_PATH)


def load_data():

    """Load data from the specified files."""

    X = np.load(os.path.join(DATA_DIR_ABS_PATH, 'X.npy'))
    y = np.load(os.path.join(DATA_DIR_ABS_PATH, 'y.npy'))
    return (X, y)


def generate_model(hps):

    """Generate and compile a TensorFlow model based on hyperparameters."""

    inputs = tf.keras.Input(shape=(hps.sequence_length, 6 + hps.fft_window_size), name='lstm_inputs')
    batch_norm = tf.keras.layers.BatchNormalization()(inputs)
    lstm1 = tf.keras.layers.LSTM(units=hps.lstm1_units, kernel_regularizer=hps.lstm1_regularizer)(batch_norm)
    outputs = tf.keras.layers.Dense(units=3)(lstm1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=hps.learning_rate), metrics=['mae'])
    return model


def draw_results(y, predictions, title):

    """Draw and save plots comparing actual and predicted values."""

    plt.figure()
    plt.plot(y[:, 0], label='actual')
    plt.plot(predictions[:, 0], label='predicted')
    plt.xlabel('day')
    plt.ylabel('%% change in price')
    plt.title(title + ' (min)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, title + '_min.png'), dpi=600, format='png')

    plt.figure()
    plt.plot(y[:, 1], label='actual')
    plt.plot(predictions[:, 1], label='predicted')
    plt.xlabel('day')
    plt.ylabel('%% change in price')
    plt.title(title + ' (avg)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, title + '_avg.png'), dpi=600, format='png')

    plt.figure()
    plt.plot(y[:, 2], label='actual')
    plt.plot(predictions[:, 2], label='predicted')
    plt.xlabel('day')
    plt.ylabel('%% change in price')
    plt.title(title + ' (max)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, title + '_max.png'), dpi=600, format='png')


def draw_history(history):

    """Draw and save plots of training history."""

    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (mse)')
    plt.title('loss')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, 'loss.png'), dpi=600, format='png')

    plt.figure()
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (mse)')
    plt.title('val loss')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, 'val_loss.png'), dpi=600, format='png')


def split_data(hps, X, y):

    """Split the data into training and validation sets."""


    index = int(hps.train_split * X.shape[0])
    return (X[:index], y[:index], X[index:], y[index:])


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR_ABS_PATH)
    hps = Hyperparameters()
    hps.save(os.path.join(RESULTS_DIR_ABS_PATH, 'hyperparameters.json'))

    with open(os.path.join(RESULTS_DIR_ABS_PATH, 'output.txt'), 'w') as f:
        X, y = load_data()
        X_train, y_train, X_val, y_val = split_data(hps, X, y)

        model = generate_model(hps)
        tf.keras.utils.plot_model(model, os.path.join(RESULTS_DIR_ABS_PATH, 'model.png'), show_shapes=True)
        model.summary()

        if (len(sys.argv) > 1):
            model.load_weights(sys.argv[1])

        history = model.fit(
            X_train,
            y_train,
            epochs=hps.epochs,
            batch_size=hps.batch_size,
            validation_data=(X_val, y_val)
        )

        model.save_weights(os.path.join(RESULTS_DIR_ABS_PATH, 'weights.h5'))

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        draw_results(y_train, train_predictions, 'train')
        draw_results(y_val, val_predictions, 'validation')
        draw_history(history)

        f.write('loss: {}\nmae: {}\nval loss: {}\nval mae: {}\n'.format(
            history.history['loss'][-1],
            history.history['mae'][-1],
            history.history['val_loss'][-1],
            history.history['val_mae'][-1]
        ))
        f.write('val rho (min): {}\nval rho (avg): {}\nval rho (max): {}'.format(
            np.corrcoef(y_val[:, 0], val_predictions[:, 0])[0, 1],
            np.corrcoef(y_val[:, 1], val_predictions[:, 1])[0, 1],
            np.corrcoef(y_val[:, 2], val_predictions[:, 2])[0, 1]
        ))
