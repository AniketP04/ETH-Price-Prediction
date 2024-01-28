import os
import json
import numpy as np
from scipy import stats
from utilities import Hyperparameters

SCRIPT_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_REL_PATH = 'data/'
RAW_DATA_DIR_REL_PATH = 'data/raw/'
DATA_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, DATA_DIR_REL_PATH)
RAW_DATA_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, RAW_DATA_DIR_REL_PATH)


def load_raw_data():

    """
    Load raw data from JSON files and process it.

    Returns:
    Tuple of NumPy arrays containing daily data for gas limit, gas price, gas used, transaction fee, 
    Ethereum market cap, and Ethereum daily price.
    """


    f_daily_avg_gas_limit = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailyavggaslimit.json'), 'r')
    f_daily_avg_gas_price = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailyavggasprice.json'), 'r')
    f_daily_gas_used = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailygasused.json'), 'r')
    f_daily_txn_fee = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailytxnfee.json'), 'r')
    f_eth_daily_market_cap = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'ethdailymarketcap.json'), 'r')
    f_eth_daily_price = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'ethdailyprice.json'), 'r')

    json_daily_avg_gas_limit = json.load(f_daily_avg_gas_limit)
    json_daily_avg_gas_price = json.load(f_daily_avg_gas_price)
    json_daily_gas_used = json.load(f_daily_gas_used)
    json_daily_txn_fee = json.load(f_daily_txn_fee)
    json_eth_daily_market_cap = json.load(f_eth_daily_market_cap)
    json_eth_daily_price = json.load(f_eth_daily_price)

    f_daily_avg_gas_limit.close()
    f_daily_avg_gas_price.close()
    f_daily_gas_used.close()
    f_daily_txn_fee.close()
    f_eth_daily_market_cap.close()
    f_eth_daily_price.close()

    daily_avg_gas_limit_data = np.asarray([[d['unixTimeStamp'], d['gasLimit']] for d in json_daily_avg_gas_limit['result']][8:], dtype=np.float64)
    daily_avg_gas_price_data = np.asarray([[d['unixTimeStamp'], d['avgGasPrice_Wei']] for d in json_daily_avg_gas_price['result']][8:], dtype=np.float64)
    daily_gas_used_data = np.asarray([[d['unixTimeStamp'], d['gasUsed']] for d in json_daily_gas_used['result']][8:], dtype=np.float64)
    daily_txn_fee_data = np.asarray([[d['unixTimeStamp'], d['transactionFee_Eth']] for d in json_daily_txn_fee['result']][8:], dtype=np.float64)
    eth_daily_market_cap_data = np.asarray([[d['unixTimeStamp'], d['marketCap']] for d in json_eth_daily_market_cap['result']][8:], dtype=np.float64)
    eth_daily_price_data = np.asarray([[d['unixTimeStamp'], d['value']] for d in json_eth_daily_price['result']][8:], dtype=np.float64)

    print(daily_avg_gas_limit_data.shape)

    return (daily_avg_gas_limit_data[:, 1], daily_avg_gas_price_data[:, 1], daily_gas_used_data[:, 1], daily_txn_fee_data[:, 1], eth_daily_market_cap_data[:, 1], eth_daily_price_data[:, 1])


def get_price_ffts(hps, eth_daily_price_data):

    """
    Calculate Fast Fourier Transform (FFT) of z-scored daily Ethereum prices.

    Args:
    hps (Hyperparameters): Object containing hyperparameters.
    eth_daily_price_data (numpy.ndarray): Array containing daily Ethereum prices.

    Returns:
    numpy.ndarray: Array containing the absolute values of the FFT of z-scored daily Ethereum prices.
    """

    windows = []
    for i in range(0, eth_daily_price_data.shape[0] - hps.fft_window_size + 1):
        window = eth_daily_price_data[i:i + hps.fft_window_size]
        windows += [stats.zscore(window)]
    windows = np.vstack(windows)
    return np.abs(np.fft.fft(windows))


def get_preprocessed_data(hps, full_sequence):

    """
    Generate preprocessed data for training the model.

    Args:
    hps (Hyperparameters): Object containing hyperparameters.
    full_sequence (numpy.ndarray): Array containing the full sequence of data.

    Returns:
    Tuple of NumPy arrays containing input data (X) and target labels (y).
    """
    
    windows, y = [], []
    for i in range(0, full_sequence.shape[0] - hps.sequence_length + 1 - hps.prediction_window_size):
        window = full_sequence[i:i + hps.sequence_length, :]
        windows += [window]
        prediction_window = full_sequence[i + hps.sequence_length:i + hps.sequence_length + hps.prediction_window_size, 5]
        y += [[
            100*(np.amin(prediction_window)/window[-1, 5]-1),
            100*(np.mean(prediction_window)/window[-1, 5]-1),
            100*(np.amax(prediction_window)/window[-1, 5]-1)
        ]]
    return (np.stack(windows), np.array(y))


if __name__ == '__main__':
    hps = Hyperparameters()

    raw_data = load_raw_data()
    eth_daily_price_data = raw_data[-1]
    price_ffts = get_price_ffts(hps, eth_daily_price_data)
    full_sequence = np.concatenate((np.stack(raw_data, axis=1)[hps.fft_window_size - 1:, :], price_ffts), axis=1)
    X, y = get_preprocessed_data(hps, full_sequence)

    print('X.shape = ', X.shape)
    print('y.shape = ', y.shape)

    np.save(os.path.join(DATA_DIR_ABS_PATH, 'X.npy'), X)
    np.save(os.path.join(DATA_DIR_ABS_PATH, 'y.npy'), y)
