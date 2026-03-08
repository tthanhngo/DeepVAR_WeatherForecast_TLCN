# DeepVAR_WeatherForecast_TLCN
Tiểu luận chuyên ngành Công nghệ Thông tin - Đề tài: Tìm Hiểu Mô Hình Deep Vector Autoregression Dùng Trong Dự Báo Thời Tiết
Sinh viên thực hiện: 
- Ngô Thanh Thanh - 21110643
- Vũ Thị Bích Ngọc - 21110905
  
Nội dung báo cáo:
1.	Tìm hiểu về thời tiết, các yếu tố ảnh hưởng và bài toán dự báo thời tiết.
2.	Tìm hiểu về các công trình liên quan dùng trong dự báo thời tiết.
3.	Tìm hiểu về chuỗi thời gian và bài toán dự báo chuỗi thời gian.
4.	Tìm hiểu lần lượt các mô hình AR, VAR, LSTM, DeepVAR.
5.	Thu thập dữ liệu về thời tiết.
6.	Cài đặt mô hình DeepVAR.
7.	Đánh giá hiệu quả mô hình trong dự báo dữ liệu thời tiết

DỮ LIỆU
-	Nguồn: meteostat.net - nền tảng cung cấp dữ liệu quan trắc thực tế từ các trạm thời tiết, tuân thủ các tiêu chuẩn của Tổ chức Khí tượng Thế giới (WMO)
-	Thời gian: thời gian 4 năm từ 07/02/2021 tới 07/02/2025, với bản ghi theo ngày
-	Địa điểm: Thủ Đức, Tokyo, Đan Mạch, St’s John
-	Tập dữ liệu gồm 8 trường:
+	date : DD/MM/YYYY
+	tavg: nhiệt độ trung bình (°C).
+	tmin: Nhiệt độ thấp nhất trong ngày (°C).
+	tmax: Nhiệt độ cao nhất ghi nhận trong ngày (°C).
+	prcp: Tổng lượng mưa trong ngày (mm).
+	wdir: Hướng gió trung bình trong ngày (°).
+	wspd: Tốc độ gió trung bình trong ngày (km/h)
+	pres: Áp suất khí quyển trung bình trong ngày (hPa).
-	Kích thước tập dữ liệu: 1462 dòng x 8 cột

TIỀN XỬ LÝ DỮ LIỆU
1.Làm sạch dữ liệu
- Xử lý các giá trị không hợp lệ
- Nội suy tuyến tính các dữ liệu thiếu
- Loại bỏ trùng lặp
2. Tăng cường dữ liệu
- Gaussian
- Numpy
3. Kiểm tra mức độ tương quan tuyến tính giữa các biến
- Sử dụng ma trận tương quan
4. Kiểm tra và xử lý tính dừng
- ADF Test
- Differencing
5. Chuẩn hóa dữ liệu
- Z-Score
- Min-Max
6. Chia tập dữ liệu
- 80% train – 20% test
- Trong 80% train tiếp tục chia 80% train - 20% validation

CÀI ĐẶT MÔ HÌNH DEEPVAR
1. Tạo dự đoán từ mô hình VAR
- Tìm best_lag của mô hình VAR dựa trên tiêu chí AIC với p chạy từ 1-->30
- Dựa trên số lag đã xác định, hàm create_var_predictions dựa trên mô hình VAR để tạo dữ liệu đầu vào cho mạng nơ-ron
2. Xây dựng mô hình DeepVAR
- Kết quả dự đoán từ mô hình VAR sẽ được chuyển đổi thành những cửa sổ trượt để làm đầu vào cho mạng LSTM
- Tìm tổ hợp siêu tham số tối ưu (best_params) cho mô hình DeepVAR bằng cách sử dụng grid search và chọn ra mô hình có MSE thấp nhất trên tập validation


ĐÁNH GIÁ THỰC NGHIỆM
1. Môi trường thực nghiệm
-	Phần cứng: Laptop/PC, intel core i5,  Windows 10 Pro 64bit, RAM 16GB, Ổ cứng SSD
-	Công cụ lập trình: 
+	Ngôn ngữ Python phiên bản 3.11.5
+	IDE: Visual Studio Code
+	Framework xây dựng giao diện người dùng:Streamlit
2. Tiêu chí đánh giá
-	Độ chính xác của mô hình được đánh giá bằng các chỉ số: MSE, MAE, RMSE, CV(RMSE)
3. Kết quả thực nghiệm
Mục tiêu dự báo của mô hình là 5 biến:
      	- tavg (nhiệt độ trung bình)
      	- pres (áp suất)
      	- prcp (lượng mưa)
      	- wspd (tốc độ gió)
      	- wdir (hướng gió)
Tỷ lệ chia tập dữ liệu:
      	- 80% cho huấn luyện (80% train – 20% validation).
      	- 20% cho kiểm tra (test).
Các bước thực nghiệm:
-	Tìm lag tối ưu
-	Tìm tham số tối ưu
-	Đánh giá độ chính xác của mô hình
Nhận xét chung:
- Mô hình DeepVAR hoạt động tốt với các biến thời tiết có tính tuần hoàn, dao động ổn định và ảnh hưởng mạnh mẽ lẫn nhau như nhiệt độ và áp suất.
- Gặp nhiều khó khăn với các biến có tính bất thường và dao động lớn như lượng mưa và hướng gió – đặc biệt ở các khu vực khí hậu nhiệt đới như Thủ Đức





