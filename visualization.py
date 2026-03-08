import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def visualize_data(data):
    """Visualize data with histograms, boxplots, or line plots."""

    # Chuyển đổi cột 'date' sang datetime nếu cần
    if 'date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

    # Đặt 'date' làm index nếu chưa
    if data.index.name != 'date':
        data = data.set_index('date')

    column = st.selectbox("Select column to visualize", data.columns)
    chart_type = st.radio("Select chart type", ["Histogram", "Boxplot", "Line Plot"])

    if chart_type == "Histogram":
        fig, ax = plt.subplots()
        ax.hist(data[column].dropna(), bins=10, color='skyblue', edgecolor='black')
        st.pyplot(fig)

    elif chart_type == "Boxplot":
        fig, ax = plt.subplots()
        sns.boxplot(x=data[column].dropna(), ax=ax, color='lightgreen')
        st.pyplot(fig)

    elif chart_type == "Line Plot":
        # Nhóm theo tháng
        grouped_data = data[column].resample('M').mean()
        date_fmt = "%m-%Y"  # Hiển thị: 01-2025, 02-2025, ...

        # Tăng kích thước biểu đồ (chiều ngang)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(grouped_data.index, grouped_data.values, color='purple')
        ax.set_title(f"{column}")
        ax.set_xlabel("Month")
        ax.set_ylabel(column)
        ax.set_xticks(grouped_data.index)
        ax.set_xticklabels([d.strftime(date_fmt) for d in grouped_data.index], rotation=45)
        st.pyplot(fig)

def visualize_column(data, column, description=None):
    if column not in data.columns:
        st.error(f"Column '{column}' not found in the dataset.")
        return

    st.subheader(f"Visualization for: {column}")
    if description:
        st.write(description)

    st.line_chart(data[column])
    
    st.write(f"Displaying {len(data[column].dropna())} non-null values from the column '{column}'.")

def plot_normalized_data(data, normalization_method):
    """Vẽ biểu đồ dữ liệu sau khi chuẩn hóa với tùy chọn cột."""
    
    # Cho người dùng chọn cột cần hiển thị
    column_to_plot = st.selectbox("Select column to display chart:", data.columns, key="norm_column_select")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data[column_to_plot], label=normalization_method, color='blue')
    ax.set_title(f"{normalization_method} Normalized Data - {column_to_plot}")
    ax.set_xlabel("Time")
    ax.set_ylabel(column_to_plot)
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

def plot_dataset_split(name, train, val, test):
    fig, ax = plt.subplots(figsize=(16, 4))

    ax.plot(train.index, train[name], label='Train', color='blue')
    ax.plot(val.index, val[name], label='Validation', color='orange')
    ax.plot(test.index, test[name], label='Test', color='green')

    ax.set_title(f'Data Split - {name}')
    ax.set_xlabel('Time')
    ax.set_ylabel(name)
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

import pandas as pd

def fill_missing_values(df, method='time', fill_value=None):

    df = df.copy()

    if method == 'time':
        # Nội suy theo thời gian
        df.interpolate(method='time', inplace=True)
    elif method == 'linear':
        df.interpolate(method='linear', inplace=True)
    elif method == 'mean':
        df.fillna(df.mean(), inplace=True)
    elif method == 'median':
        df.fillna(df.median(), inplace=True)
    elif method == 'mode':
        df.fillna(df.mode().iloc[0], inplace=True)
    elif method == 'constant':
        if fill_value is None:
            raise ValueError("You need to provide a fill_value when selecting method='constant'")
        df.fillna(fill_value, inplace=True)
    else:
        raise ValueError(f"Method '{method}' is not supported.")

    return df

# So sánh dữ liệu tăng cường và dữ liệu gốc
def compare_original_augmented(dataset, dataset_aug, column, figsize=(10, 8)):
    if column not in dataset.columns or column not in dataset_aug.columns:
        raise ValueError(f"Column '{column}' does not exist in one of the DataFrames.")
    plt.figure(figsize=figsize)

    # Biểu đồ 1: So sánh toàn bộ dữ liệu
    plt.subplot(2, 1, 1)
    plt.plot(dataset[column], label="Original", alpha=0.7)
    plt.plot(dataset_aug[column], label="Augmented", alpha=0.5, linestyle='--')
    plt.legend()
    plt.title(f"Compare original data and augmented data - {column}")

    # Biểu đồ 2: Chỉ phần dữ liệu mới (trước thời gian của dataset gốc)
    plt.subplot(2, 1, 2)
    augmented_part = dataset_aug.loc[dataset_aug.index < dataset.index.min()]
    plt.plot(augmented_part.index, augmented_part[column], label="Generated Data", linestyle='--', color='red')
    plt.legend()
    plt.title(f"Only augmented data - {column}")

    plt.tight_layout()

    st.pyplot(plt)  # HIỂN THỊ biểu đồ trong Streamlit

# Vẽ biểu đồ phân phối giá trị các trường
def plot_distribution(data, column, bins=30, figsize=(10,6)):
    if column not in data.columns:
        raise ValueError(f"Column '{column}' does not exist in one of the DataFrames.")

    plt.figure(figsize=figsize)
    sns.histplot(data[column], bins=bins, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()

    st.pyplot(plt)  # HIỂN THỊ biểu đồ trong Streamlit

# Vẽ biểu đồ đường quan sát sự thay đổi theo thời gian
def plot_smoothed_time_series(dataset, column, window=7, title=None, xlabel="Date", ylabel=None, figsize=(10, 5)):
    if column not in dataset.columns:
        raise ValueError(f"Column '{column}' does not exist in one of the DataFrames.")
    
    smoothed = dataset[column].rolling(window=window, center=True).mean()
    
    plt.figure(figsize=figsize)
    plt.plot(dataset.index, smoothed, label=f'{window}-day Rolling Mean', color='orange')
    plt.plot(dataset.index, dataset[column], label='Original', alpha=0.3)  # gốc mờ
    plt.title(title if title else f"Smoothed Change in {column} through date")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else column)
    plt.legend()
    plt.tight_layout()

    st.pyplot(plt)  # HIỂN THỊ biểu đồ trong Streamlit

# Vẽ biểu đồ loss
def plot_loss_curve(loss_df):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_df["Epoch"], loss_df["Training Loss"], label="Training Loss")
    plt.plot(loss_df["Epoch"], loss_df["Validation Loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    st.pyplot(plt)  # HIỂN THỊ biểu đồ trong Streamlit


def plot_actual_vs_predicted_streamlit(y_test, predictions, variable_index=0, step_index=0, variable_name=None):
    
    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:, step_index, variable_index], label='Actual')
    plt.plot(predictions[:, step_index, variable_index], label='Predicted')

    var_label = variable_name if variable_name else f"Variable {variable_index}"
    plt.title(f"Actual vs Predicted - {var_label} (Look-ahead step {step_index + 1})")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    st.pyplot(plt)  

# Chuẩn hoá dữ liệu bằng min-max    
def min_max_normalize (min,max,dataset):
    scaler = MinMaxScaler(feature_range=(min, max))
    scaled_data = pd.DataFrame(
                scaler.fit_transform(dataset),
                columns=dataset.columns,
                index=dataset.index,
            )
    return scaled_data, scaler