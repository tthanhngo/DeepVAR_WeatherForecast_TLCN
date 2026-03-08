# Standard libraries
import base64
import json
import os
import pickle
import threading
import time

import matplotlib.pyplot as plt

# Deep learning with GluonTS and MXNet
import numpy as np

# Data manipulation and visualization
import pandas as pd
import seaborn as sns

# Streamlit for UI
import streamlit as st
import tensorflow as tf

# Machine learning and preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from tensorflow.keras.models import load_model

from model import (
    build_deepvar,
    create_var_predictions,
    create_windows,
    evaluate_multivariate_forecast,
    evaluate_overall_forecast,
    find_bestlag,
    grid_search,
)

# Import custom modules
from preprocessing import (
    augment_with_gaussian,
    augment_timeseries_data,
    check_stationarity,
    compute_correlation_matrix,
    preprocess_data,
    split_train_test,
    split_train_val,
    make_stationary,
)
from visualization import (
    compare_original_augmented,
    fill_missing_values,
    min_max_normalize,
    plot_actual_vs_predicted_streamlit,
    plot_normalized_data,
    plot_smoothed_time_series,
    visualize_data,
    plot_dataset_split,
)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{{"png"}};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(page_title="Deep Vector Autoregression")
    st.title("Weather Indicators Forecasting Tool")
    add_bg_from_local("Background/Home.jpg")

    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
    if uploaded_file:

        # 1. Load data

        file_name = uploaded_file.name[:-4]

        directory_results = f"results/{file_name}"
        directory_models = f"models/{file_name}"

        if not os.path.exists(directory_results):
            os.makedirs(directory_results)

        if not os.path.exists(directory_models):
            os.makedirs(directory_models)

        data = pd.read_csv(uploaded_file)

        st.write("### Raw Data")
        st.dataframe(data)
        
        st.markdown(
            "<h1 style='font-size: 30px; color: black;'>Visualization</h1>",
            unsafe_allow_html=True,
        )
        with st.expander("➡ Click to view chart"):
            visualize_data(data)
        data = preprocess_data(data)
        data = fill_missing_values(data, method="mean")
        compute_correlation_matrix(data)

        # 4. Thực hiện Augmentation dữ liệu

        # Checkbox cho Augmentation

        st.sidebar.write(
            "<h2 style='font-size: 30px; color: black;'>Augmentation</h2>",
            unsafe_allow_html=True,
        )

        # Tạo danh sách lựa chọn phương pháp tăng cường
        augment_method = st.sidebar.selectbox(
            "Choose Augmentation Method:",
            ["Gaussian", "Numpy"]
        )

        augment_option = st.sidebar.checkbox("Augment Data")

        if augment_option:

            # Áp dụng phương pháp tương ứng
            if augment_method == "Gaussian":
                stddev = 0.05
                mean = 0.0
                dataset_aug = augment_with_gaussian(data, mean, stddev)
            elif augment_method == "Numpy":
                dataset_aug = augment_timeseries_data(data, len(data))

            st.markdown(
            "<h1 style='font-size: 30px; color: black;'>Augmentation</h1>",
            unsafe_allow_html=True,        )

            # Hiển thị thông tin dữ liệu
            st.write("Original data size:", data.shape)
            st.write("Augmented dataset size:", dataset_aug.shape)

            # Hiển thị tập dữ liệu tăng cường
            st.dataframe(dataset_aug)
            
            with st.expander("➡ View comparison chart"):

                # Chọn cột dữ liệu để so sánh
                column_to_compare = st.selectbox("Select a data column", data.columns)

                # Gọi hàm so sánh với cột đã chọn
                compare_original_augmented(data, dataset_aug, column_to_compare)

            # Gộp dữ liệu gốc với dữ liệu tăng cường
            data = pd.concat([data, dataset_aug], axis=0).reset_index(drop=True)

        st.markdown(
            "<h1 style='font-size: 30px; color: black;'>Stationarity test results</h1>",
            unsafe_allow_html=True,
        )
        with st.expander("➡ View results"):
            # 5. Kiểm tra tính dừng trước khi chuẩn hóa
            st.sidebar.markdown(
            "<h1 style='font-size: 30px; color: black;'>Check Stationarity</h1>",
            unsafe_allow_html=True,
            )
            #visualize_column(data, selected_col)

            # Step 4: Stationary check
            if st.sidebar.checkbox("Check Stationarity"):
                stationarity_results = check_stationarity(data)
                st.subheader("Stationarity Check Results")
                st.dataframe(stationarity_results)

            # Step 5: Make data stationary if needed
            if st.sidebar.checkbox("Make Data Stationary"):
                data = make_stationary(data)
                st.subheader("Stationary Data")
                stationarity_results = check_stationarity(data)
                st.dataframe(stationarity_results)
                #st.dataframe(data)

                #visualize_column(
                #    data, selected_col, description="(After making stationary)"
                #)  
            
        data.drop(["tmin", "tmax"], axis=1, inplace=True)

        # Xem lại dữ liệu sau khi dừng hoá
        st.write("Data table")
        st.write(data)

        st.markdown(
            "<h1 style='font-size: 30px; color: black;'>Data column chart.</h1>",
            unsafe_allow_html=True,
        )
        with st.expander("➡ View results"):
            plot_smoothed_time_series(data, column="tavg", window=14)
            plot_smoothed_time_series(data, column="pres", window=14)
            plot_smoothed_time_series(data, column="prcp", window=14)
            plot_smoothed_time_series(data, column="wspd", window=14)
            plot_smoothed_time_series(data, column="wdir", window=14)

        # 2. Chuẩn hóa

        # Chọn phương pháp chuẩn hóa
        st.sidebar.write(
            "<h2 style='font-size: 30px; color: black;'>Normalization</h2>",
            unsafe_allow_html=True,
        )
        normalization_method = st.sidebar.radio(
            "**Select Data Normalization Method:**",
            ["No Normalization", "Min-Max Normalization", "Z-Score Normalization"],
        )

        if normalization_method == "Min-Max Normalization":
            val_input = st.sidebar.text_input(
                "Enter (minimum, maximum) values", "0,1"
            )
            min_val, max_val = map(float, val_input.split(","))
            scaler = MinMaxScaler(feature_range=(min_val, max_val))
        elif normalization_method == "Z-Score Normalization":
            scaler = StandardScaler()
        else:
            scaler = None

        if scaler:
            scaled_data = pd.DataFrame(
                scaler.fit_transform(data),
                columns=data.columns,
                index=data.index,
            )
        else:
            scaled_data = data

        data = scaled_data

        # 3. Cho phép người dùng chọn cách hiển thị dữ liệu

        st.markdown(
            "<h1 style='font-size: 30px; color: black;'>View normalized data</h1>",
            unsafe_allow_html=True,
        )

        with st.expander("➡ Click to view data"):
            view_option = st.radio(
                "Select how to view normalized data:",
                ("View Chart", "View Data Table"),
            )

            if view_option == "View Chart":
                plot_normalized_data(data, normalization_method)
            else:
                st.dataframe(data)

        # train_data, test_data = split_data(data)
        # st.write(f"Train: {len(train_data)}, Test: {len(test_data)}")

        # 6. Chia tập dữ liệu

        st.sidebar.write(
            "<h2 style='font-size: 30px; color: black;'>Train - Test - Val Split</h2>",
            unsafe_allow_html=True,
        )

        data_length = len(data)
        st.sidebar.header("Data Splitting")
        train_test_ratio = st.sidebar.slider(
            "Train-Test Split Ratio", 0.1, 0.9, 0.8
        )
        test_ratio = 1 - train_test_ratio
        st.sidebar.text(
            f"Train: {train_test_ratio:.1f}, Test: {test_ratio:.1f}"
        )
        final_train_data, test_data = split_train_test(data, train_test_ratio)

        st.markdown(
            "<h1 style='font-size: 30px; color: black;'>Dataset splitting results</h1>",
            unsafe_allow_html=True,
        )

        st.write(
            f"Train-Test Split: Train: {len(final_train_data)} rows, "
            f"Test: {len(test_data)} rows"
        )

        # Train-Validation Split Ratio (within Train)
        train_val_ratio = st.sidebar.slider(
            "Train-Validation Split Ratio",
            0.1,
            0.9,
            0.2,
            key="train_val_slider",
        )
        val_ratio = 1 - train_val_ratio
        st.sidebar.text(
            f"Validation: {train_val_ratio:.1f}, Train: {val_ratio:.1f}"
        )
        final_train_data, validation_data = split_train_val(
            final_train_data, train_val_ratio
        )
        if len(final_train_data) == 0 or len(validation_data) == 0:
            st.error("Invalid ratio. Please split again!")
        else:
            st.write(
                f"Train-Validation Split: Train: {len(final_train_data)} rows,"
                f" Validation: {len(validation_data)} rows"
            )

        st.write("### Test:")
        st.dataframe(test_data)
        st.write("### Train:")
        st.dataframe(final_train_data)
        st.write("### Validation:")
        st.dataframe(validation_data)

        with st.expander("➡ View Chart"):
            plot_dataset_split('tavg', final_train_data, validation_data, test_data)
            plot_dataset_split('prcp', final_train_data, validation_data, test_data)
            plot_dataset_split('pres', final_train_data, validation_data, test_data)
            plot_dataset_split('wspd', final_train_data, validation_data, test_data)
            plot_dataset_split('wdir', final_train_data, validation_data, test_data)

        st.write(f"Total number of samples: {data_length}")
        st.write(f"Train: {len(final_train_data)} samples")
        st.write(f"Validation: {len(validation_data)} samples")
        st.write(f"Test: {len(test_data)} sample")

        model_type = st.sidebar.selectbox(
            "**Select Model:**",
            [
                "DEEPVAR"
            ],
        )

        grid_search_path = f"results/{file_name}/DeepVAR_grid_search_results.json"
        training_history_path = f"results/{file_name}/training_history.pkl"   
       
        # 7. Training

        st.sidebar.subheader("Model Training")
        train_button = st.sidebar.button("Train Model")
        stop_button = st.sidebar.button("Stop Training")
        test_button = st.sidebar.button("Test Model")

        # Hiển thị thông báo ở trung tâm màn hình thay vì sidebar
        status_placeholder = st.empty()

        stop_training = threading.Event()

        # Callback to Stop Training
        class StopTrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if stop_training.is_set():
                    self.model.stop_training = True
                    st.warning(f"Training stopped at epoch {epoch + 1}")

        if stop_button:
            stop_training.set()
            st.warning("Stopping training process...")

        global flag
        flag = False

        if train_button and not stop_training.is_set():
            stop_training.clear()
            
            spinner_html = '''
                <div style="display: flex; align-items: center;">
                    <div class="loader"></div>
                    <p style="margin-left: 10px; font-size: 30px">🚀 Optimizing parameters...</p>
                </div>
                <style>
                    .loader {
                        border: 4px solid #f3f3f3;
                        border-top: 4px solid #3498db;
                        border-radius: 50%;
                        width: 55px; height: 55px;
                        animation: spin 1s linear infinite;
                    }

                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
                '''

            status_placeholder.markdown(spinner_html, unsafe_allow_html=True)

            st.write("### 📂 Model training results")

            status_placeholder.write("### Optimizing parameters...")

            best_lag = find_bestlag(final_train_data, 31)
            var = VAR(endog=final_train_data.values)
            var_result = var.fit(maxlags=best_lag)
            # var_result.aic

            train_var_pred = create_var_predictions(
                final_train_data, var_result, var_result.k_ar, data.columns
            )
            val_var_pred = create_var_predictions(
                validation_data, var_result, var_result.k_ar, data.columns
            )

            # Chuyển đổi dữ liệu
            look_back = best_lag
            look_ahead = 1

            X_train = create_windows(
                train_var_pred, window_shape=look_back, end_id=-look_ahead
            )
            y_train = create_windows(
                final_train_data.values[var_result.k_ar :],
                window_shape=look_ahead,
                start_id=look_back,
            )

            X_val = create_windows(
                val_var_pred, window_shape=look_back, end_id=-look_ahead
            )
            y_val = create_windows(
                validation_data.values[var_result.k_ar :],
                window_shape=look_ahead,
                start_id=look_back,
            )

            # st.write(X_train.shape, y_train.shape)
            # st.write(X_val.shape, y_val.shape)

            # Tìm tham số tối ưu
            param_grid = {
                "learning_rate": [7e-4, 1e-3, 3e-3],
                "batch_size": [64, 256, 512],
                "units_lstm": [128, 96, 64],  # số nơ ron mạng LSTM
                "epoch": [70, 120, 160],
            }

            best_params, best_mse, search_time = grid_search(
                X_train,
                y_train,
                X_val,
                y_val,
                param_grid,
                look_back,
                look_ahead,
            )

            st.write("Parameter optimization completed.")
            st.write("**Best Parameters:**")
            st.json(best_params)
            st.write(f"**Best MSE:** {best_mse:.4f}")
            st.write(f"**Search Time:** {search_time:.2f} seconds")
            st.write(f"**Best Lags (VAR):** {var_result.k_ar}")

            with open(f"results/{file_name}/var_result.pkl", "wb") as f:
                pickle.dump(var_result, f)

            grid_search_results = {
                "best_parameters": best_params,
                "best_mse": best_mse,
                "search_time": search_time,
                "best_lag": best_lag,
                "look_back": look_back,
                "look_ahead": look_ahead,
            }

            with open(
                f"results/{file_name}/DeepVAR_grid_search_results.json",  # noqa: E501
                "w",
            ) as f:
                json.dump(grid_search_results, f)

            # Training
            final_model = build_deepvar(
                input_dim=X_train.shape[2],
                output_dim=y_train.shape[2],
                look_back=look_back,
                look_ahead=look_ahead,
                lr=best_params["learning_rate"],
                units_lstm=best_params["units_lstm"],
            )

            start_time = time.time()
            es = tf.keras.callbacks.EarlyStopping(
                patience=10,
                verbose=1,
                min_delta=0.001,
                monitor="val_loss",
                mode="auto",
                restore_best_weights=True,
            )

            try:
                history = final_model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=best_params["epoch"],
                    batch_size=best_params["batch_size"],
                    verbose=1,
                    callbacks=[StopTrainingCallback(),]
                )
                end_time = time.time()

                train_time = end_time - start_time
                st.write(f"Training time: {train_time:.2f} seconds")

                with open(
                    f"results/{file_name}/training_history.pkl", "wb"
                ) as f:
                    pickle.dump(history.history, f)

                if not stop_training.is_set():
                    final_model.save(
                        f"models/{file_name}/DeepVAR_final_model.keras"
                    )
                    st.write("The model has been saved.")

            except Exception as e:
                st.error(f"An error occurred during training: {e}")

            if stop_training.is_set():
                st.warning("Training process has been stopped by the user.")

            else:
                # Tạo DataFrame chứa loss theo từng epoch
                loss_df = pd.DataFrame(
                    {
                        "Epoch": range(
                            1, len(history.history["loss"]) + 1
                        ),  # noqa: E501
                        "Training Loss": history.history["loss"],
                        "Validation Loss": history.history["val_loss"],
                    }
                )

                st.line_chart(loss_df.set_index("Epoch"))
                st.write("### Final Losses")
                st.write(
                    f"Final Training Loss: {history.history['loss'][-1]:.4f}"  # noqa: E501
                )
                st.write(
                    f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}"  # noqa: E501
                )

        # 8. Stop #

        if stop_training.is_set():

            grid_searchs_path = (
                f"results/{file_name}/DeepVAR_grid_search_results.json"
            )
            training_history_path = f"results/{file_name}/training_history.pkl"

            if not os.path.exists(grid_searchs_path) or not os.path.exists(
                training_history_path
            ):
                st.error(
                    "❌ The complete result files could not be found. Please train the model first."
                )
            else:
                with open(
                    f"results/{file_name}/DeepVAR_grid_search_results.json",
                    "r",
                ) as f:
                    grid_search_data = json.load(f)
                
                st.write("### Last Saved Results")
                st.json(grid_search_data)
                
                with open(
                    f"results/{file_name}/training_history.pkl", "rb"
                ) as f:
                    loaded_history = pickle.load(f)

                if "val_loss" in loaded_history:
                    loss_df = pd.DataFrame(
                        {
                            "Epoch": range(1, len(loaded_history["loss"]) + 1),
                            "Training Loss": loaded_history["loss"],
                            "Validation Loss": loaded_history["val_loss"],
                        }
                    )
                else:
                    loss_df = pd.DataFrame(
                        {
                            "Epoch": range(1, len(loaded_history["loss"]) + 1),
                            "Training Loss": loaded_history["loss"],
                        }
                    )                
                # Vẽ biểu đồ Loss theo Epoch
                st.line_chart(loss_df.set_index("Epoch"))

                # Hiển thị giá trị cuối cùng
                st.write("### Final Losses")
                st.write(f"Final Training Loss: {loaded_history['loss'][-1]:.4f}")
                st.write(f"Final Validation Loss: {loaded_history['val_loss'][-1]:.4f}")

        # 9. Test #

        if test_button:

            st.write("### 📂 Model testing results")

            # Đường dẫn các file cần thiết
            grid_search_path = (
                f"results/{file_name}/DeepVAR_grid_search_results.json"
            )
            var_result_path = f"results/{file_name}/var_result.pkl"
            model_path = f"models/{file_name}/DeepVAR_final_model.keras"

            if (
                not os.path.exists(grid_search_path)
                or not os.path.exists(var_result_path)
                or not os.path.exists(model_path)
            ):
                st.error(
                    "❌ The complete result files could not be found. Please train the model first."
                )
            else:
                with open(
                    f"results/{file_name}/DeepVAR_grid_search_results.json",
                    "r",
                ) as f:
                    grid_search_results = json.load(f)

                best_lag = grid_search_results.get("best_lag", None)
                search_time = grid_search_results.get("search_time", None)
                look_back = grid_search_results.get("look_back", None)
                look_ahead = grid_search_results.get("look_ahead", None)

                try:
                    with open(f"results/{file_name}/var_result.pkl", "rb") as f:
                        var_result = pickle.load(f)
                except FileNotFoundError:
                    st.error(
                        "❌ The complete result files could not be found. Please train the model first."  # noqa: E501
                    )
                    var_result = None

                start_time = time.time()

                deepvar_model = load_model(
                    f"models/{file_name}/DeepVAR_final_model.keras"
                )

                test_var_pred = create_var_predictions(
                    test_data, var_result, best_lag, test_data.columns
                )

                X_test = create_windows(
                    test_var_pred, window_shape=look_back, end_id=-look_ahead
                )

                y_test = create_windows(
                    test_data.values[var_result.k_ar :],
                    window_shape=look_ahead,
                    start_id=look_back,
                )

                predictions = deepvar_model.predict(X_test)

                execution_time = time.time() - start_time

                evaluation_df = evaluate_multivariate_forecast(
                    y_test, predictions, data.columns
                )
                evaluation_df_overall = evaluate_overall_forecast(
                    y_test, predictions, execution_time
                )

                st.write(evaluation_df)
                st.write(evaluation_df_overall)

                plot_actual_vs_predicted_streamlit(
                    y_test,
                    predictions,
                    variable_index=0,
                    step_index=0,
                    variable_name="tavg",
                )
                plot_actual_vs_predicted_streamlit(
                    y_test,
                    predictions,
                    variable_index=1,
                    step_index=0,
                    variable_name="prcp",
                )
                plot_actual_vs_predicted_streamlit(
                    y_test,
                    predictions,
                    variable_index=2,
                    step_index=0,
                    variable_name="wdir",
                )
                plot_actual_vs_predicted_streamlit(
                    y_test,
                    predictions,
                    variable_index=3,
                    step_index=0,
                    variable_name="wspd",
                )
                plot_actual_vs_predicted_streamlit(
                    y_test,
                    predictions,
                    variable_index=4,
                    step_index=0,
                    variable_name="pres",
                )

        # 10. Predict #
        st.markdown("### 📂 Weather dataset prediction (.csv)")
        uploaded_file = st.file_uploader("Select CSV file", type=["csv"])

        if uploaded_file:
            st.write("Dataset has been uploaded:")
            weather_recent = pd.read_csv(uploaded_file)
            cols_to_drop = ["tmin", "tmax", "snow", "wpgt", "tsun"]
            weather_recent.drop(
                columns=weather_recent.columns.intersection(cols_to_drop),
                inplace=True
            )
            weather_recent = preprocess_data(weather_recent)
            st.write(weather_recent)

            var_result_path = f"results/{file_name}/var_result.pkl"
            grid_search_path = f"results/{file_name}/DeepVAR_grid_search_results.json"
            model_path = f"models/{file_name}/DeepVAR_final_model.keras"
            
            # Hiển thị nút Predict sau khi có file
            predict_button = st.button("### 🚀 Prediction")

            if predict_button:
                if (not os.path.exists(var_result_path) or not os.path.exists(grid_search_path) or not os.path.exists(model_path)):
                    st.error(
                        "❌ The complete result files could not be found. Please train the model first."  # noqa: E501
                    )
                    var_result = None
                else:
                    with open(
                        f"results/{file_name}/DeepVAR_grid_search_results.json",
                        "r",
                    ) as f:
                        grid_search_results = json.load(f)
                        
                    best_lag = grid_search_results.get("best_lag", None)
                    search_time = grid_search_results.get("search_time", None)
                    look_back = grid_search_results.get("look_back", None)
                    look_ahead = grid_search_results.get("look_ahead", None)

                    with open(f"results/{file_name}/var_result.pkl", "rb") as f:
                        var_result = pickle.load(f)

                    # Bước 2: Chuẩn hoá dữ liệu
                    weather_recent_scaled, scaler = min_max_normalize(
                        0, 1, weather_recent
                    )
            
                    deepvar_model = load_model(
                        f"models/{file_name}/DeepVAR_final_model.keras"
                    )              
                        
                    var_pred = create_var_predictions(
                        weather_recent_scaled,
                        var_result,
                        best_lag,
                        weather_recent_scaled.columns,
                    )
                    var_pred_input = create_windows(
                        var_pred, window_shape=look_back, end_id=-look_ahead
                    )
                    predictions = deepvar_model.predict(var_pred_input)

                    predictions = scaler.inverse_transform(
                        predictions.reshape(-1, predictions.shape[-1])
                    ).reshape(predictions.shape)
                    # predictions shape: (1, n_days, num_features)
                    n_days = predictions.shape[1]

                    # Tạo các ngày tiếp theo dựa trên ngày cuối cùng của dữ liệu
                    last_date = weather_recent.index[-1]
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=n_days,
                        freq="D",
                    )

                    # predictions[0]: lấy ra chuỗi (n_days, num_features)
                    predicted_df = pd.DataFrame(
                        predictions[0],
                        columns=weather_recent.columns,
                        index=future_dates,
                    )
                    predicted_df.index.name = "Date"

                    # In kết quả
                    st.write("📅 Weather forecast for the next day")
                    st.write(predicted_df)


if __name__ == "__main__":
    main()
