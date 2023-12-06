# Standard python libraries
import os

# Installed libraries
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pickle

# Imports from our package
from lightautoml.tasks import Task
from lightautoml.addons.autots.base import AutoTS
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.pipelines.features.lgb_pipeline import LGBSeqSimpleFeatures
from lightautoml.pipelines.features.linear_pipeline import LinearTrendFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.reader.base import DictToPandasSeqReader
from lightautoml.automl.blend import WeightedBlender
from lightautoml.ml_algo.random_forest import RandomForestSklearn

from moexalgo import Market, Ticker

# Disable warnings
import warnings
warnings.filterwarnings("ignore")


HORIZON = 1
TARGET_COLUMN = "close"
DATE_COLUMN = "date"

#Настройки
ticker_name = 'ALRS'
period = '1D'
date_from = '2007-07-20'
date_to = '2023-12-05'

# Акции
ticker = Ticker(ticker_name)

# Свечи по акциям за период. Период в минутах 1, 10, 60 или '1m', '10m', '1h', 'D', 'W', 'M', 'Q'; по умолчанию 60
df = ticker.candles(date=date_from, till_date=date_to, period=period)

df.drop(['end'], axis=1, inplace=True)
df.rename(columns={"begin": "date"}, inplace=True)

# Построение стратегии анализа с помо
MyStrategy = ta.Strategy(
    name="DCSMA10",
    ta=[
        {"kind": "ohlc4"},
        {"kind": "sma", "length": 3},
        {"kind": "sma", "length": 5},
        {"kind": "sma", "length": 10},
        {"kind": "sma", "length": 30},
        {"kind": "sma", "length": 50},
        {"kind": "donchian", "lower_length": 5, "upper_length": 5},
        {"kind": "ema", "close": "OHLC4", "length": 5, "suffix": "OHLC4"},
        {"kind": "ema", "close": "OHLC4", "length": 15, "suffix": "OHLC4"},
        {'kind': 'log_return', 'cumulative': True, 'append': True},
 {'kind': 'rsi', 'append': True},
 # {'kind': 'macd', 'append': True},
 # {'kind': 'mad', 'append': True},
 # {'kind': 'massi', 'append': True},
 # {'kind': 'mcgd', 'append': True},
 # {'kind': 'psar', 'append': True},
 # {'kind': 'qstick', 'append': True},
 # {'kind': 'quantile', 'append': True},
 # {'kind': 'rma', 'append': True},
 # {'kind': 'roc', 'append': True},
 #  {'kind': 'rsx', 'append': True},
 # {'kind': 'stoch', 'append': True},
 # {'kind': 'tsignals', 'append': True},
 # # {'kind': 'ttm_trend', 'append': True},
 # {'kind': 'xsignals', 'append': True},
 # # {'kind': 'zlma', 'append': True},
 # {'kind': 'zscore', 'append': True}
     ]
)

df.ta.strategy(MyStrategy, append=True)

# Сделаем временной ряд непрерывным и смержим с датасетом
date = pd.DataFrame(pd.date_range(start=date_from, end=date_to), columns=['date'])
df = date.merge(df, how='left', on='date')

# Заполним пропуски с помощью интерполяции
for col in df.columns:
    if col == 'date':
        continue
    df[col] = df[col].interpolate(method='linear')

# Удалим строки с пустыми ячейками индикаторов
df = df.dropna()
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

test_start = df[DATE_COLUMN].values[-HORIZON]

train = df[df[DATE_COLUMN] < test_start].copy()
test = df[df[DATE_COLUMN] >= test_start].copy()

task = Task("multi:reg", greater_is_better=False, metric="mae", loss="mae")

roles = {
    "target": TARGET_COLUMN,
    DatetimeRole(seasonality=('m', 'wd'), base_date=True): DATE_COLUMN,
}

seq_params = {
    "seq0": {
        "case": "next_values",
        "params": {
            "n_target": HORIZON,
            "history": HORIZON,
            "step": 1,
            "from_last": True,
            "test_last": True
        }
    }
}

transformers_params = {
    "lag_features": 20,
    "lag_time_features": 20,
    "diff_features": [1, 2, 3, 4, 5, 6, 7, 14],
}

### Параметры для модели тренда.
trend_params = {
    'trend': True,
    'train_on_trend': True,
    'trend_type': 'decompose',  # one of 'decompose', 'decompose_STL', 'linear' or 'rolling'
    'trend_size': 3,
    'decompose_period': 3,
    'detect_step_quantile': 0.01,
    'detect_step_window': 1,
    'detect_step_threshold': 0.7,
    'rolling_size': 1,
    'verbose': 0
}

automl = AutoTS(
    task,
    reader_params = {
        "seq_params": seq_params
    },
    rf_params={"default_params":{"criterion":"squared_error"}},
    time_series_trend_params=trend_params,
    time_series_pipeline_params=transformers_params,
    config_path=r"C:\Users\tdall\anaconda3\envs\py39\Lib\site-packages\lightautoml\automl\presets\time_series_config.yml"
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

univariate_train_pred, _ = automl.fit_predict(train, roles, verbose=4)
forecast, _ = automl.predict(train)

# Выведем результаты и MAE
print('pred:', forecast, "\n", "real:", test.close.values)
print(f"MAE: {mean_absolute_error(test.close, forecast)}")

# Отправить результат на сервер
result =  [{
        "ticker": ticker_name,
        "predict_price": str(forecast[-1]),
        "predict_profit": str(forecast[-1]/test.close.values[-1] - 1),
        "timeframe": HORIZON
    }]

count = 0
while True:
    try:
        url = 'http://213.171.14.97:8080/api/v1/leaderboard'
        response = requests.post(url, json=result)
        if response.status_code == 200:
            print("Запрос успешно отправлен:")
            break
    except Exception as err:
        print("Ошибка отправка запроса на API:", err)

    # Делаем повторные попытки в случае ошибки
    if count >= 5:
        break

    count += 1