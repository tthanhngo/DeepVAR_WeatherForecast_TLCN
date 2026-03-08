import numpy as np
import time
import threading
import pandas as pd
import os
import random
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tqdm import tqdm
from statsmodels.tsa.vector_ar.var_model import VAR

stop_training = threading.Event()

class StopTrainingCallback:
    def __init__(self):
        self.stop_training = False
    
    def check_stop(self):
        return stop_training.is_set()

# Tìm lag tốt nhất
def find_bestlag (train_data, range_lag):
    
    results = []
    AIC = {}
    best_aic, best_lag = np.inf, 0
    
    for i in tqdm(range(range_lag)):
        model = VAR(endog = train_data.values)
        model_result = model.fit(maxlags=i)
        AIC[i] = model_result.aic
            
        results.append([i, AIC[i]])
        if AIC[i] < best_aic:
            best_aic = AIC[i]
            best_lag = i
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['p', 'AIC']
    result_df = result_df.sort_values(by='p', ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(14,5))
    plt.plot(range(len(AIC)), list(AIC.values()))
    plt.plot([best_lag], [best_aic], marker='o', markersize=8, color="red")
    ticks = list(range(0, len(AIC), 1))
    labels = [str(i) for i in ticks]
    plt.xticks(ticks, labels, rotation=90)
    plt.xlabel('lags')
    plt.ylabel('AIC')

    # Hiển thị biểu đồ trên Streamlit
    st.pyplot(fig)

    # Hiển thị bảng kết quả
    st.write(result_df)

    return best_lag

# Tạo dự đoán từ mô hình VAR
def create_var_predictions(data, model, lag_order, features):
    lagged_data = []
    for i in range(lag_order, len(data)):        
        pred = model.forecast(data[features].values[i-lag_order:i], steps=1)
        lagged_data.append(pred[0])
    
    return np.array(lagged_data)

# Tạo cửa sổ trượt
def create_windows(data, window_shape, step=1, start_id=None, end_id=None):
    data = np.asarray(data)
    data = data.reshape(-1, 1) if np.prod(data.shape) == np.max(data.shape) else data

    start_id = 0 if start_id is None else start_id
    end_id = data.shape[0] if end_id is None else end_id

    data = data[int(start_id):int(end_id), :]
    window_shape = (int(window_shape), data.shape[-1])
    step = (int(step),) * data.ndim
    slices = tuple(slice(None, None, st) for st in step)
    indexing_strides = data[slices].strides
    win_indices_shape = ((np.array(data.shape) - window_shape) // step) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(data.strides))

    window_data = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=strides)
    
    return np.squeeze(window_data, 1)

# Gieo số ngẫu nhiên
def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

# Xây dựng mô hình deepvar (var_lstm)
def build_deepvar(input_dim, output_dim, look_back, look_ahead, lr, units_lstm):
    
    """
    input_dim: số đặc trưng (n_features)
    output_dim: số đầu ra tại mỗi bước (n_outputs)
        - 'lr': learning rate
        - 'units_lstm': số units của LSTM
        - 'look_back': số bước nhìn lại (sequence length đầu vào)
        - 'look_ahead': số bước dự đoán (sequence length đầu ra)
    """
    set_seed(33)

    input_shape = (look_back, input_dim)
    inp = Input(shape=input_shape)

    x = LSTM(units_lstm, activation='tanh')(inp)
    x = RepeatVector(look_ahead)(x)
    x = LSTM(units_lstm, activation='tanh', return_sequences=True)(x)
    out = TimeDistributed(Dense(output_dim))(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    return model

# Tìm tham số tối ưu
def grid_search(train_var_pred, train_targets, val_var_pred, val_targets, param_grid, look_backk, look_aheadd):
    
    best_param = None
    best_mse = float("inf")
    start_time = time.time()

    if train_var_pred.shape[0] != train_targets.shape[0]:
        raise ValueError("Mismatch between training features and targets dimensions.")
    if val_var_pred.shape[0] != val_targets.shape[0]:
        raise ValueError("Mismatch between validation features and targets dimensions.")
    
    if not all(key in param_grid for key in ['learning_rate', 'batch_size', 'units_lstm', 'epoch']):
        raise ValueError("Parameter grid must contain 'learning_rate', 'batch_size', 'units_lstm', and 'epoch'.")
    es = tf.keras.callbacks.EarlyStopping(patience=30, verbose=0, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)

    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            for units_lstm in param_grid['units_lstm']:
                for epoch in param_grid['epoch']:
                    model = build_deepvar(input_dim=train_var_pred.shape[2], 
                                        output_dim=train_targets.shape[2], 
                                        look_back = look_backk ,
                                        look_ahead = look_aheadd,
                                        lr=lr,
                                        units_lstm=units_lstm)
                    model.fit(
                        train_var_pred, train_targets,
                        epochs=epoch,
                        batch_size=batch_size,
                        validation_data=(val_var_pred, val_targets),
                        verbose=0,
                        callbacks=[es]
                    )

                    val_var_lstm_pred = model.predict(val_var_pred)
                    # Để tính MSE trên toàn bộ dữ liệu cần flatten (làm phẳng) toàn bộ mảng
                    mse = mean_squared_error(val_targets.flatten(), val_var_lstm_pred.flatten())
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'units_lstm': units_lstm,
                            'epoch': epoch
                        }

    execution_time = time.time() - start_time

    return best_params, best_mse, execution_time

# Kết quả đánh giá cho từng biến
def evaluate_multivariate_forecast(y_test, predictions, column_names):
    mse_dict = {}
    rmse_dict = {}
    mae_dict = {}
    cv_rmse_dict = {}

    for i, col in enumerate(column_names):
        mse = mean_squared_error(y_test[..., i], predictions[..., i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[..., i], predictions[..., i])
        y_mean = np.mean(y_test[..., i])
        cv_rmse = rmse / y_mean if y_mean != 0 else np.nan

        mse_dict[col] = mse
        rmse_dict[col] = rmse
        mae_dict[col] = mae
        cv_rmse_dict[col] = cv_rmse

    evaluation_df = pd.DataFrame({
        'Variable': column_names,
        'MSE': pd.Series(mse_dict),
        'RMSE': pd.Series(rmse_dict),
        'MAE': pd.Series(mae_dict),
        'CV_RMSE': pd.Series(cv_rmse_dict)
    })

    return evaluation_df

def evaluate_overall_forecast(y_test, predictions, execution_time=None):
    test_mse = mean_squared_error(y_test.flatten(), predictions.flatten())
    test_mae = mean_absolute_error(y_test.flatten(), predictions.flatten())
    test_rmse = np.sqrt(test_mse)
    y_mean = np.mean(y_test.flatten())
    cv_rmse = test_rmse / y_mean if y_mean != 0 else np.nan

    metric_names = [
        "Test Time (seconds)",
        "Test MSE",
        "Test MAE",
        "Test RMSE",
        "Test CV RMSE"
    ]
    metric_values = [
        f"{execution_time:.2f}" if execution_time is not None else "N/A",
        f"{test_mse:.4f}",
        f"{test_mae:.4f}",
        f"{test_rmse:.4f}",
        f"{cv_rmse:.4f}",
    ]

    evaluation_df_overall = pd.DataFrame({
        "Metric": metric_names,
        "Value": metric_values
    })

    return evaluation_df_overall