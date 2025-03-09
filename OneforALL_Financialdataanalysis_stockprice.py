import yfinance as yf#Yahoo finance lấy giá chứng khoán
import pandas as pd#Đọc dữ liệu
import xlwings as xw#Xử lý excel
import numpy as np#Xử lý dữ liệu
import matplotlib.pyplot as plt#Vẽ biểu đồ
from matplotlib.dates import YearLocator, DateFormatter, MonthLocator#Formatting dates trên trục biểu đồ
from sklearn.preprocessing import MinMaxScaler#Chuẩn hóa dữ liệu
from keras.callbacks import ModelCheckpoint#Lưu lại dữ liệu
from tensorflow.keras.models import load_model#Cải mô hình
#Các lớp để xây dựng mô hình
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
#Kiểm tra độ chính xác của mô hình
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

###Tải giá chứng khoán
data = yf.download(tickers='ADS.DE', period='10y', interval='1d')
print(data)

###Lưu giá chứng khoán vào file excel
wb = xw.Book()
ws = wb.sheets["Sheet1"]
ws["A1"].options(pd.DataFrame, header=1, index=True, expand='table').value = data
wb.save('Adidasstock.xlsx')
wb.close()

###Đọc file excel và tiền xử lý
df = pd.read_excel("Adidasstock.xlsx")
df1 = pd.DataFrame(df, columns=["Adj Close"])
df1.index = df["Date"]
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df["Year"] = df["Date"].dt.year
Datetime = df.pop("Year")
df.insert(1, "Year", Datetime)


###Biểu đồ giá thực tế
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df['Adj Close'], label='Giá thực tế', color='red')
plt.xlabel('Thời gian')
plt.ylabel('giá đóng cửa(VND)')
plt.title('Biểu đồ giá đóng cửa')
plt.legend(loc='best')
years = YearLocator()
yearsFmt = DateFormatter('%Y')
months = MonthLocator()#Thêm dòng này để khai báo MonthLocator
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(yearsFmt)
plt.gca().xaxis.set_minor_locator(months)
plt.tight_layout()
plt.show()


###Chia tập dữ liệu thành tập training và tập test
#cho df1 thành numpy và gán thành data
data = df1.values
#chia dữ liệu thành 80% training 20% test
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

###Chuẩn hóa dữ liệu
sc = MinMaxScaler(feature_range=(0, 1))
sc_train = sc.fit_transform(data)

###Chuẩn bị tập training
x_train, y_train = [], []
for i in range(50, len(train_data)):#chạy for từ 50 đến hết train_data
    x_train.append(sc_train[i-50:i, 0])#x_train lấy giá trị từ idx 0-50,1-51,2-52
    y_train.append(sc_train[i, 0])#y_train lấy giá trị từ idx 51,52,53
x_train = np.array(x_train)#đổi x_train thành numpy array
y_train = np.array(y_train)#đổi y_train thành numpy array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))#đổi shape x-train thành kiểu phù hợp với model

###Xây dựng LSTM model
model = Sequential()#khởi tạo Sequential
model.add(LSTM(units=128, input_shape=(x_train.shape[1], 1), return_sequences=True))#thêm lớp LSTM vào model
model.add(LSTM(units=64))#thêm lớp LSTM khác vào model
model.add(Dropout(0.5))#thêm lớp Dropout để giảm overfitting
model.add(Dense(1))#Dự đoán 1 giá trị(giá đóng cửa)
model.compile(loss='mean_absolute_error', optimizer='adam')#compile model
###Train the model
save_model = 'save_model_hdf5.keras'#tên mô hình
#Lưu trữ mô hình
best_model = ModelCheckpoint(save_model, monitor='loss', verbose=2, save_best_only=True, mode='auto')
#Huấn luyện mô hình, thiết lập các thông số huấn luyện
model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=2, callbacks=[best_model])
###Load the best model
final_model = load_model('save_model_hdf5.keras')

###Dự đoán tập training
y_train_predict = final_model.predict(x_train)#dự đoán x-train
y_train_predict = sc.inverse_transform(y_train_predict.reshape(-1, 1))#biến y-train predict trở lại giá trị thực
y_train = sc.inverse_transform(y_train.reshape(-1, 1))#biến y-train trở lại giá trị thực

###Chuẩn bị tập dữ liệu test
test = df1[len(train_data)-50:].values
test = test.reshape(-1, 1)
sc_test = sc.transform(test)
x_test = []
for i in range(50, test.shape[0]):
    x_test.append(sc_test[i-50:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

###Dự đoán tập test
y_test_predict = final_model.predict(x_test)
y_test_predict = sc.inverse_transform(y_test_predict.reshape(-1, 1))
y_test = data[train_size:]

###Tạo tập dữ liệu với cùng len(trian_data) và len(test_data)
train_data1 = df1[50:train_size]
test_data1 = df1[train_size:]
train_data1['Dự đoán'] = y_train_predict#thêm dữ liệu dự đoán tập train vào cột dự đoán của tập train_data1
test_data1['Dự đoán'] = y_test_predict#thêm dữ liệu dự đoán tập test vào cột dự đoán của tập test_data1
df.loc[50:train_size-1, "dự đoán train"] = train_data1["Dự đoán"].values#chèn giá trị dự đoán train vào df
df.loc[train_size:, "dự đoán test"] = test_data1["Dự đoán"].values#chèn giá trị dự đoán test vào df

#Biểu đồ giá thực tế, giá dự đoán train và test
plt.figure(figsize=(24,8))
plt.plot(df["Date"], df["Adj Close"],label='Giá thực tế',color='red')#đường giá thực
plt.plot(df["Date"], df["dự đoán train"],label='Giá dự đoán train',color='green')#đường giá dự báo train
plt.plot(df["Date"], df["dự đoán test"],label='Giá dự đoán test',color='blue')#đường giá dự báo test
plt.xlabel('Thời gian')#đặt tên trục x
plt.ylabel('Giá đóng cửa (VNĐ)')#đặt tên trục y
plt.title('So sánh giá thực tế, giá dự đoán train và giá dự đoán test')#đặt tên biểu đồ
plt.legend(loc='best')
years = YearLocator()
yearsFmt = DateFormatter('%Y')
months = MonthLocator()  # Thêm dòng này để khai báo MonthLocator
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(yearsFmt)
plt.gca().xaxis.set_minor_locator(months)
plt.tight_layout()
plt.show()

# r2
print("Độ phù hợp tập train:", r2_score(y_train, y_train_predict))
# mae
print("Sai số tuyệt đối trung bình train:", mean_absolute_error(y_train, y_train_predict))
# mape
print("Phần trăm sai số tuyệt đối trung bình tập train:", mean_absolute_percentage_error(y_train, y_train_predict))
# r2
print('Độ phù hợp test:', r2_score(y_test, y_test_predict))
# mae
print("Sai số tuyệt đối trung bình tập test:", mean_absolute_error(y_test, y_test_predict))
# mape
print("Phần trăm sai số tuyệt đối trung bình tập test:", mean_absolute_percentage_error(y_test, y_test_predict))

#################################################################
###Dự đoán giá đóng cửa ngày tiếp theo:
##Tính ngày kế tiếp
next_date = df["Date"].iloc[-1] + pd.Timedelta(days=1)
##Chuyển kiểu dữ liệu ngày kế tiếp thành datetime
next_date = pd.to_datetime(next_date)
##Dự đoán ngày kế tiếp
#Lấy 50 giá cuối cùng để train
x_test1 = np.array([sc_train[-50:, 0]])
x_test1 = np.reshape(x_test1,(x_test1.shape[0], x_test1.shape[1], 1))
y_test1_predict = final_model.predict(x_test1)
y_test1_predict = sc.inverse_transform(y_test1_predict)
##Lấy giá của ngày cuối cùng
last_closing_price = df["Adj Close"].iloc[-1]
#Giá dự đoán ngày kế tiếp
df_nextdate_predict = pd.DataFrame({"Date":[next_date], "Giá dự đoán":[y_test1_predict[0][0]], "Giá ngày trước": [last_closing_price]})
print(df_nextdate_predict)