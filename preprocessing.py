import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(dataset, drop_threshold=1):
    # Chuyển đổi trường Date sang kiểu dữ liệu datetime để có thể sử dụng trong các phép toán thời gian.
    dataset['date'] = pd.to_datetime(dataset['date'], format='%d/%m/%Y') 
    # Đặt cột Date làm chỉ mục
    dataset.set_index('date', inplace=True)
    # Chuyển đổi dữ liệu thành dạng số, các giá trị không thể chuyển thành số sẽ bị thay bằng NaN
    for col in dataset.columns:
        if dataset[col].dtype == object: # correct type
            dataset[col] = pd.to_numeric(dataset[col].str.replace(',', '.'))
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    # Loại bỏ các cột có quá nhiều giá trị khuyết thiếu
    missing_ratios = dataset.isnull().mean()  # Tính tỷ lệ thiếu dữ liệu trên mỗi cột
    columns_to_drop = missing_ratios[missing_ratios > drop_threshold].index # Xác định các cột có tỷ lệ thiếu > 1
    dataset.drop(columns=columns_to_drop, inplace=True) # Loại bỏ các cột đó
    # Nội suy tuyến tính dựa trên khoảng cách thời gian
    dataset.interpolate(method='time', inplace=True)
    # Điền nốt giá trị còn thiếu nếu có
    dataset.fillna(dataset.mean(), inplace=True)  
    # Loại bỏ các hàng có chỉ mục NaN
    dataset = dataset[~dataset.index.isnull()]
    # Loại bỏ các dòng trùng lặp
    dataset.drop_duplicates(inplace=True)
    return dataset

def make_stationary(data, lag=1):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    if not non_numeric_columns.empty:
        print(f"Warning: Non-numeric columns excluded: {list(non_numeric_columns)}")

    numeric_data = data.select_dtypes(include=['number'])
    differenced_data = numeric_data.apply(difference_series, lag=lag)

    return differenced_data

def compute_correlation_matrix(data):
    """Compute and visualize the correlation matrix."""
    st.write("<h2 style='font-size: 20px; color: black;'>Correlation matrix between variables</h2>", unsafe_allow_html=True)
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, cbar=True)
    ax.set_title('Correlation matrix of variables in the dataset')
    st.pyplot(fig)

def adf_test(series, title=''):
    series_clean = series.dropna()
    result = adfuller(series_clean)
    
    return {
        'Column': title,
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Stationary': 'Yes' if result[1] < 0.05 else 'No'
    }

def check_stationarity(data):
    results = []
    
    for column in data.columns:
        result = adf_test(data[column], title=column)
        results.append(result)

    return pd.DataFrame(results)
"""
def check_stationarity(data):
    
    for col in data.select_dtypes(include=[np.number]).columns:
        result = adfuller(data[col].dropna())
        print(f"{col}: ADF Statistic = {result[0]}, p-value = {result[1]}")
"""

def normalize_data(data, method):
    """Normalize data using MinMax or Z-Score scaling."""
    if method == "MinMax":
        scaler = MinMaxScaler()
    elif method == "Z-Score":
        scaler = StandardScaler()
    else:
        return data
    data[data.columns] = scaler.fit_transform(data[data.columns])
    return data

def split_data(data, train_ratio=0.8):
    """Split data into training and testing sets."""
    train_size = int(len(data) * train_ratio)
    train_data, test_data = data[:train_size], data[train_size:]
    return train_data, test_data

def split_train_test(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    return train_data, test_data

def split_train_val(train_data, val_ratio):
    val_size = int(len(train_data) * val_ratio)
    val_data = train_data.iloc[:val_size]
    train_data = train_data.iloc[val_size:]
    
    return train_data, val_data

# Phát sinh dữ liệu
# Thêm nhiễu Gaussian (phân phối chuẩn) vào một chuỗi thời gian.
def add_gaussian_noise(time_series, mean, stddev):
    noise = np.random.normal(mean, stddev, len(time_series))  # Tạo nhiễu Gaussian
    noisy_series = time_series + noise  # Thêm nhiễu vào dữ liệu gốc
    return noisy_series


# Sinh tập chỉ mục thời gian mới với số lượng periods bằng số lượng dòng của dữ liệu gốc
def generate_new_dates(df, periods):
    first_date = df.index.min()  # Lấy ngày nhỏ nhất trong dữ liệu gốc
    return pd.date_range(end=first_date - pd.DateOffset(days=1),
                         periods=periods,
                         freq='D')  # Sinh danh sách ngày mới, cách nhau 1 ngày

# Tạo một bản sao dữ liệu gốc có nhiễu Gaussian và ghép vào dữ liệu ban đầu
# stddev = 0.1 * data.std()  # Chọn 10% độ lệch chuẩn của dữ liệu gốc
# mean=0.0
def augment_with_gaussian(data, mean, stddev): 
    augmented_datasets = []
    augmented_data = data.copy()
        
    for column in data.columns:
        if np.issubdtype(data[column].dtype, np.number):  # Kiểm tra cột có phải là số không
            augmented_data[column] = add_gaussian_noise(
                data[column].dropna(), mean, stddev)
        
    new_dates = generate_new_dates(data, len(data))  # Sinh index mới
    augmented_data.index = new_dates  # Gán index mới cho dữ liệu có nhiễu
    augmented_datasets.append(augmented_data)
        
    return pd.concat([data] + augmented_datasets, axis=0).sort_index()  # Hợp nhất và sắp xếp index


# Phát sinh dữ liệu mới bằng phương pháp Numpy dựa trên xu hướng (trend), mùa vụ (seasonality) và nhiễu Gaussian (noise)
def augment_timeseries_data(data, n_periods):    
    new_dates = generate_new_dates(data, n_periods)
    augmented_data = []
    
    for column in data.columns:
        # Tính toán thống kê của dữ liệu gốc
        mean = data[column].mean()
        std = data[column].std()
        trend = np.polyfit(range(len(data)), data[column].values, 1)[0]
        
        # Tạo thành phần xu hướng
        base_trend = -np.arange(n_periods)[::-1] * trend + mean
        # Tạo thành phần mùa vụ
        seasonality = np.sin(np.linspace(0, 2*np.pi, 12)) * std * 0.5
        # Thêm nhiễu Gaussian
        noise = np.random.normal(0, std * 0.1, n_periods)
        
        # Tạo chuỗi thời gian mới
        # Chuỗi mới có xu hướng, mùa vụ và nhiễu giống dữ liệu gốc
        new_series = pd.Series(
            base_trend + np.tile(seasonality, n_periods//12 + 1)[:n_periods] + noise,
            index=new_dates
        )
        # Ghép dữ liệu mới vào dữ liệu gốc & sắp xếp lại theo thời gian.
        augmented_data.append(new_series)
    
    return (pd.concat([pd.DataFrame(dict(zip(data.columns, augmented_data))), data])).sort_index()

# Thực hiện dừng hóa dữ liệu.

# Lấy sai phân bậc 1
def difference_series(series, lag=1):
    return series.diff(periods=lag).dropna()

def make_stationary(data, lag=1):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    exclude_cols = ['date']  # Các cột cần loại bỏ khỏi xử lý

    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    if not non_numeric_columns.empty:
        print(f"Warning: Non-numeric columns excluded: {list(non_numeric_columns)}")

    numeric_data = data.select_dtypes(include=['number']).drop(columns=[col for col in exclude_cols if col in data.columns])

    differenced_columns = {}
    for col in numeric_data.columns:
        differenced_series = numeric_data[col].diff(periods=lag).dropna()
        differenced_series.index = range(len(differenced_series))  # Tránh lỗi trùng index
        differenced_columns[col] = differenced_series

    differenced_data = pd.DataFrame(differenced_columns)

    # Ghép lại cột 'date' (nếu tồn tại) với dữ liệu đã sai phân
    if 'date' in data.columns:
        # Cắt bớt phần đầu tương ứng với độ trễ lag
        date_series = data['date'].iloc[lag:].reset_index(drop=True)
        differenced_data.insert(0, 'date', date_series)

    return differenced_data