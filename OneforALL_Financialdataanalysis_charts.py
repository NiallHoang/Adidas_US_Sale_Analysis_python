
# khai báo thư viện
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind




# đọc dữ liệu
Adidas_data = pd.read_excel("Adidas US Sales Datasets.xlsx")


# Đổi tên các cột và xóa ký tự
Adidas_data_re = Adidas_data.rename(columns={
    'Retailer ID': 'Retailer_ID',
    'Invoice Date': 'Invoice_Date',
    'Price per Unit': 'Price_per_Unit',
    'Units Sold': 'Units_Sold',
    'Total Sales': 'Total_Sales',
    'Operating Profit': 'Operating_Profit',
    'Operating Margin': 'Operating_Margin',
    'Sales Method': 'Sales_Method'
}).copy()


# KIỂM ĐỊNH

#Lọc dữ liệu và tính ma trận tương quan cho các biến số học.
df_numerical = Adidas_data.iloc[:, 7:12]
corr_matrix = df_numerical.corr()
alpha = 0.05 # thiết lập mức ý nghĩa 0.05
str_filtered = Adidas_data[Adidas_data["Product"].str.contains("Street Footwear")] # lọc các hàng chứa chuỗi street footwear
apparel_filtered = Adidas_data[Adidas_data["Product"].str.contains("Apparel")] # lọc các hàng chứa chuỗi Apparel
athletic_filtered = Adidas_data[Adidas_data["Product"].str.contains("Athletic Footwear")] # # lọc các hàng chứa chuỗi Athletic Footwear

###Kiểm định independent sample t-test trung bình units sold của region Northeast và Southeast liệu có khác nhau
#Vì thấy tổng Operateting Profit của South và Southeast gần tương đương nhau nên:
Adidas_data.groupby(by="Region").agg({"Operating Profit": "sum"}) # tính tổng lợi nhuận mỗi vùng

print("H0: trung bình units sold của 2 regions Northeast và Southeast như nhau") # in ra giả thuyết
print("H1: trung bình units sold của 2 regions Northeast và Southeast khác nhau") # in ra giả thuyết

unit_Northeast = Adidas_data[Adidas_data["Region"]=="Northeast"]["Units Sold"].tolist() # lọc các hàng trong tệp dữ liệu là Northeast , chuyển Units Sold thành dạng list
unit_Southeast = Adidas_data[Adidas_data["Region"]=="Southeast"]["Units Sold"].tolist() # lọc các hàng trong tệp dữ liệu là Southeast , chuyển Units Sold thành dạng list

t_statistic1, p_value1 = ttest_ind(unit_Northeast, unit_Southeast) # kiểm định độc lập so sánh 2 giá trị
print(f"t_stastistic: {t_statistic1}") # in giá trị thống kê
print(f"p_value: {p_value1}") # in giá trị kiểm định
if p_value1 < alpha:
    print("Bác bỏ giả thuyết H0-->Chấp nhận H1: trung bình units sold của 2 regions khác nhau")
else:
    print("Chấp nhận giả thuyết H0: trung bình units sold của 2 regions như nhau")




###Kiểm định indenpendent sample t-test trung bình giá của men street footwear và nữ có khác nhau không
# lọc các hàng có Product bằng Men's Street Footwear, chuyển Price per Unit thành dạng list
price_menstr = str_filtered[str_filtered["Product"]=="Men's Street Footwear"]["Price per Unit"].tolist()
# lọc các hàng có Product bằng Women's Street Footwear, chuyển Price per Unit thành dạng list
price_womenstr = str_filtered[str_filtered["Product"]=="Women's Street Footwear"]["Price per Unit"].tolist()
# kiểm định độc lập so sánh 2 giá trị
t_statistic2, p_value2 = ttest_ind(price_menstr, price_womenstr)
print("H0: trung bình giá của Street Footwear Nam và Nữ là giống nhau")
print("H1: trung bình giá của Street Footwear Nam và Nữ là khác nhau")
print(f"t_stastistic: {t_statistic2}") # in ra giá trị thống kê
print(f"p_value: {p_value2}") # in ra gái trị kiểm định
if p_value2 < alpha:  
    print("Bác bỏ giả thuyết H0-->Chấp nhận H1: trung bình giá của Street Footwear Nam và Nữ là khác nhau")
else:
    print("Chấp nhận giả thuyết H0: trung bình giá của Street Footwear Nam và Nữ là giống nhau")



###Kiểm định independent sample t-test trung bình units sold của region Northeast và Southeast liệu có khác nhau
#Vì thấy tổng Operateting Profit của South và Southeast gần tương đương nhau nên
Adidas_data.groupby(by="Region").agg({"Operating Profit": "sum"})

print("H0: trung bình units sold của 2 regions Northeast và Southeast như nhau")
print("H1: trung bình units sold của 2 regions Northeast và Southeast khác nhau")

unit_Northeast = Adidas_data[Adidas_data["Region"]=="Northeast"]["Units Sold"].tolist()
unit_Southeast = Adidas_data[Adidas_data["Region"]=="Southeast"]["Units Sold"].tolist()

t_statistic1, p_value1 = ttest_ind(unit_Northeast, unit_Southeast)
print(f"t_stastistic: {t_statistic1}")
print(f"p_value: {p_value1}")
if p_value1 < alpha:
    print("Bác bỏ giả thuyết H0-->Chấp nhận H1: trung bình units sold của 2 regions khác nhau")
else:
    print("Chấp nhận giả thuyết H0: trung bình units sold của 2 regions như nhau")




###Kiểm định indenpendent sample t-test trung bình giá của men street footwear và nữ có khác nhau không
price_menstr = str_filtered[str_filtered["Product"]=="Men's Street Footwear"]["Price per Unit"].tolist()
price_womenstr = str_filtered[str_filtered["Product"]=="Women's Street Footwear"]["Price per Unit"].tolist()
t_statistic2, p_value2 = ttest_ind(price_menstr, price_womenstr)
print("H0: trung bình giá của Street Footwear Nam và Nữ là giống nhau")
print("H1: trung bình giá của Street Footwear Nam và Nữ là khác nhau")
print(f"t_stastistic: {t_statistic2}")
print(f"p_value: {p_value2}")
if p_value2 < alpha:
    print("Bác bỏ giả thuyết H0-->Chấp nhận H1: trung bình giá của Street Footwear Nam và Nữ là khác nhau")
else:
    print("Chấp nhận giả thuyết H0: trung bình giá của Street Footwear Nam và Nữ là giống nhau")





##Biểu đồ tương quan Units Sold và Total Sales
# sử dụng hàm pearsonr từ thư viện scipy.stats để tính hệ số tương quan và giá trị 2 biến Units Sold , Total Sales
r_corr1, p_value3 = stats.pearsonr(df_numerical["Units Sold"].values, df_numerical["Total Sales"].values)
print("Hệ số tương quan Pearson:", r_corr1) # in hệ số tương quan
print("Giá trị p:", p_value3) # in gái trị p
if p_value3 < alpha:
    print("Có mối tương quan tuyến tính có ý nghĩa thống kê giữa hai biến.")
else:
    print("Không có mối tương quan tuyến tính có ý nghĩa thống kê giữa hai biến.")
# sử dụng hàm corrcoef của thư viện numpy để tính ma trận 
corr_sale_unitsolds = np.corrcoef(df_numerical["Units Sold"], df_numerical["Total Sales"])[0][1]
print(f"Hệ số tương quan của Units Sold và Total Sales: {corr_sale_unitsolds}") #hệ số tương quan
sns.set(style="whitegrid") # thiết lập biểu đô sử dụng seaborn
plt.figure(figsize=(9, 9)) # tạo figure 
# tạo biểu đồ tán sắc (scatterplot) 
scatter1 = sns.scatterplot(data=df_numerical, x="Units Sold", y="Total Sales", palette='Blues', sizes=(20, 200), legend=False)
# thêm đường hồi quy tuyến tính 
sns.regplot(data=df_numerical, x="Units Sold", y="Total Sales", scatter=False, color="red")
plt.xlabel("Units Sold")
plt.ylabel("Total Sales")
plt.title("Correlation of Units Sold and Total Sales")
plt.show()




#Tương quan giữa Total Sales và Operating Profit
#Biểu đồ tương quan của Total Sales và Operating Profit
# sử dụng hàm pearsonr từ thư viện scipy.stats để tính hệ số tương quan và giá trị 2 biến Total Sales, Operating Profit
r_corr2, p_value4 = stats.pearsonr(df_numerical["Total Sales"].values, df_numerical["Operating Profit"].values)
print("Hệ số tương quan Pearson:", r_corr2) 
print("Giá trị p:", p_value4)
if p_value4 < alpha:
    print("Có mối tương quan tuyến tính có ý nghĩa thống kê giữa hai biến.")
else:
    print("Không có mối tương quan tuyến tính có ý nghĩa thống kê giữa hai biến.")
# sử dụng hàm corrcoef của thư viện numpy để tính ma trận 
corr_sale_operating = np.corrcoef(df_numerical["Total Sales"], df_numerical["Operating Profit"])[0][1]
print(f"Hệ số tương quan của Total Sales và Operating Profit: {corr_sale_operating}") #Hệ số tương quan
sns.set(style="whitegrid")
plt.figure(figsize=(9, 9))
scatter2 = sns.scatterplot(data=df_numerical, x="Total Sales", y="Operating Profit", cmap="coolwarm", sizes=(20, 200), legend=False)
sns.regplot(data=df_numerical, x="Total Sales", y="Operating Profit", scatter=False, color="green")
plt.xlabel("Total Sales")
plt.ylabel("Operating Profit")
plt.title("Correlation of Total Sales and Operating Profit ")
plt.show()





#Kiểm định Chi-Square 2 biến Sales Method và Region có phụ thuộc với nhau
# tạo 1 dataframe chứa các cột và sử dụng drop_duplicates() để loia5 bỏ các giá trị trùng lặp
unique_df_chi2 = Adidas_data[["Region", "Retailer", "City", "State", "Sales Method"]].drop_duplicates()
print("H0: 2 biến Sales Method và Region độc lập với nhau")
print("H1: 2 biến Sales Method và Region phụ thuộc với nhau")
# Sử dụng hàm crosstab của pandas để tạo bảng chéo thể hiện tần số xuất hiện của 2 biến
contingency_table = pd.crosstab(unique_df_chi2["Sales Method"], unique_df_chi2["Region"])
# Sử dụng hàm chi2_contingency từ thư viện scipy.stats để thực hiện kiểm định
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Bảng chéo giữa Sales Method và Region:\n", contingency_table)
print("\nGiá trị Chi-Square:", chi2)
print("P-value:", p)
print("Bậc tự do:", dof)
print("Bảng tần số kỳ vọng:\n", expected)
if p < alpha:
    print("Bác bỏ giả thuyết H0-->Chấp nhận H1: 2 biến Sales Method và Region phụ thuộc với nhau")
else:
    print("Chấp nhận giả thuyết H0: 2 biến Sales Method và Region độc lập với nhau")




# PHÂN TÍCH

# SO SÁNH TOTAL SALES 2020 2021
# Tạo cột Year từ cột Invoice_Date
Adidas_data["Year"] = pd.to_datetime(Adidas_data["Invoice Date"]).dt.year
# Tính tổng doanh thu của năm 2020 và 2021
sumsale2020 = Adidas_data[Adidas_data["Year"] == 2020]["Total Sales"].sum()
sumsale2021 = Adidas_data[Adidas_data["Year"] == 2021]["Total Sales"].sum()
# Tính tổng doanh thu theo năm
sumsale20_21 = Adidas_data.groupby("Year")["Total Sales"].sum().reset_index()
# Tạo biểu đồ cột
plt.figure(figsize=(5, 5))
figer = sns.barplot(x="Year", y="Total Sales", data = sumsale20_21)
figer.set_yticks([0, 200000000, 400000000, 600000000, 800000000, 1000000000])
figer.set_yticklabels(['0', '200M', '400M', '600M', '800M', '1000M',])
# Thiết lập nhãn trục x và y
plt.title(f'Total Sales by Years')
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
# Thêm nhãn cho các thanh trong biểu đồ
for container in figer.containers:
    figer.bar_label(container, fmt='%.0f')
# Hiển thị biểu đồ
plt.show()



# LẤY 5 THÀNH PHỐ CÓ DOANH THU CAO NHẤT THEO NĂM
# extract year  
# Tạo cột year bằng cách trích năm từ cột Invoice Date
Adidas_data['Year'] = Adidas_data['Invoice Date'].dt.year 
Adidas_data['Year'] # in ra
# Gom dữ liệu theo cột "Year" và "City", sau đó tính tổng doanh thu ("Total Sales") cho mỗi nhóm.
city_sales = Adidas_data.groupby(['Year', 'City'])['Total Sales'].sum().reset_index()
# Nhóm dữ liệu city_sales theo cột "Year", sử dụng hàm lambda với apply để lọc 5 thành phố cao nhất, sử dụng reset_index(drop=True) để thiết lập lại chỉ số  
top_cities_sales = city_sales.groupby('Year').apply(lambda x: x.nlargest(5, 'Total Sales')).reset_index(drop=True)
# Vẽ biểu đồ bar cho top 5 thành phố theo từng năm
plt.figure(figsize=(15, 10))
for i, year in enumerate(top_cities_sales['Year'].unique(), 1): # duyệt từng năm , đánh số thứ tự từ 1
    plt.subplot(2, 2, i) # tạo biểu đồ subplot kích thước 2x2 tương ứng mỗi năm
    # lọc dữ liệu để lấy các hàng trong top_cities_sales của năm hiện tại 
    top_cities_year = top_cities_sales[top_cities_sales['Year'] == year]
    figer = sns.barplot(x='City', y='Total Sales', data=top_cities_year, palette='Accent')
    # Thiết lập điểm đánh dấu
    figer.set_yticks([0, 10000000, 20000000, 30000000, 40000000])
    figer.set_yticklabels(['0', '10M', '20M', '30M', '40M',])
    # thêm chú thích vào mỗi thanh với tổng doanh thu
    for p in figer.patches:
        figer.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)
    plt.title(f'Top 5 Cities by Total Sales - {year}')
    plt.ylabel('Total Sales')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    

# SO SÁNH VỀ CÁCH BÁN HÀNG
# Gom dữ liệu theo cột "Sales_Method" và "Total_Sales", sau đó tính tổng doanh thu 
sales_by_method = Adidas_data_re.groupby('Sales_Method')['Total_Sales'].sum().reset_index()
# Tạo biểu đồ
plt.figure(figsize=(10, 6))
figer = sns.barplot(x='Sales_Method', y='Total_Sales', data=sales_by_method, palette='magma')
# thêm chú thích vào mỗi thanh với tổng doanh thu
for index, row in sales_by_method.iterrows():
    figer.text(index, row.Total_Sales, f"${row.Total_Sales/1e6:.1f}M", color='black', ha="center")
# Thiết lâp điểm dánh dấu
yticks = [0, 100000000, 200000000, 300000000, 400000000]
yticklabels = ['0', '100M', '200M', '300M', '400M']
figer.set_yticks(yticks)
figer.set_yticklabels(yticklabels)
plt.title('Total Sales by Sales Method')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()


# TẠO BẢNG OPERATING PROFIT BY RETAILERS
# Gom dữ liệu theo cột "Retailer" và "Operating_Profit", sau đó tính tổng doanh thu 
profit_by_retailer = Adidas_data_re.groupby('Retailer')['Operating_Profit'].sum().reset_index()
# sắp xếp dữ liệu giảm dần
profit_by_retailer = profit_by_retailer.sort_values(by='Operating_Profit', ascending=False)
# tạo bảng
plt.figure(figsize=(10, 6))
bars = plt.barh(profit_by_retailer['Retailer'], profit_by_retailer['Operating_Profit'], color='dodgerblue')
plt.xlabel('Profit (M)')
plt.title('Operating Profit by Retailers')
plt.gca().invert_yaxis()  # đảo ngược trục y để thanh lớn nhất xuất hiện ở đầu 
# thêm chú thích từng thanh
for bar in bars:
    width = bar.get_width()
    plt.text(width - 5, bar.get_y() + bar.get_height() / 2, f'{width:.1f}M', va='center', ha='right', color='white', fontsize=10, fontweight='bold')
plt.show()


# tạo bảng sale per season
# lấy tháng ra từ Invoice Date
Adidas_data['Month'] = Adidas_data['Invoice Date'].dt.month
Adidas_data['Month']

# lấy năm ra từ Invoice Date 
Adidas_data['Year'] = Adidas_data['Invoice Date'].dt.year
Adidas_data['Year']

# Lấy ngày ra từ Invoice Date
Adidas_data['Day'] = Adidas_data['Invoice Date'].dt.day
Adidas_data['Day']

# Tạo Cột mùa

def find_seasons(monthNumber):
    if monthNumber in [12, 1, 2]:
        return 'Winter'
    
    elif monthNumber in [3, 4, 5]:
        return 'Spring'
    
    elif monthNumber in [6, 7, 8]:
        return 'Summer'
    
    elif monthNumber in [9, 10, 11]:
        return 'Autumn'
    

# Định nghĩa hàm find_seasons để xác định mùa từ số tháng
Adidas_data['Season'] = Adidas_data['Month'].apply(find_seasons)
Adidas_data['Season']


# chuyển đổi giá trị số của tháng sang tên tháng
Adidas_data['Month'] = pd.to_datetime(Adidas_data['Month'], format='%m').dt.month_name()

# loại cột  Invoice Date, Retailer ID 
Adidas_data.drop(columns = ['Retailer ID', 'Invoice Date'], inplace = True)

# chuyển đổi kiểu dữ liệu của cột
Adidas_data["Day"] = Adidas_data['Day'].astype('category')
# -----------------
Adidas_data['Season'] = Adidas_data['Season'].astype('category')
# -----------------
Adidas_data['Year'] = Adidas_data['Year'].astype('category')

# Tổng hợp doanh thu theo năm
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
# Đặt thứ tự mùa theo season_order
Adidas_data['Season'] = pd.Categorical(Adidas_data['Season'], categories=season_order, ordered=True)
# Nhóm dữ liệu cột season và year, tính tổng doanh thu
sales_by_season_year = Adidas_data.groupby(['Season', 'Year'])['Total Sales'].sum().reset_index()
graph = sns.catplot(x="Season", y="Total Sales", col="Year", data=sales_by_season_year, kind="bar", ci=None)
for ax in graph.axes.flatten():
    ax.set_yticks([0, 50000000, 100000000, 150000000, 200000000, 250000000])
    ax.set_yticklabels(['0', '50M', '100M', '150M', '200M', '250M'])
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 500000,  # điều chỉnh chiều cao
                f'{int(height)}',  # hiện thị số dữ liệu thô
                ha="center") # canh chỉnh dữ liệu giữa cột
graph.set_axis_labels("Season", "Total Sales")
plt.show()


# DOANH SỐ THEO THÁNG
# Trích xuất tháng và năm từ cột Invoice Date
Adidas_data_re['Month'] = Adidas_data_re['Invoice_Date'].dt.month  # Trích xuất tháng từ cột Invoice_Date
Adidas_data_re['Year'] = Adidas_data_re['Invoice_Date'].dt.year  # Trích xuất năm từ cột Invoice_Date
# Định nghĩa thứ tự của các tháng
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']  # Định nghĩa thứ tự các tháng
# Chuyển đổi cột Month sang tên tháng và thiết lập thứ tự
Adidas_data_re['Month'] = pd.Categorical(
    Adidas_data_re['Invoice_Date'].dt.strftime('%B'),
    categories=month_order,
    ordered=True
)  # Chuyển đổi cột Month sang tên tháng và thiết lập thứ tự

# Tổng hợp doanh số theo tháng
sales_by_month = Adidas_data_re.groupby(['Year', 'Month'])['Total_Sales'].sum().reset_index()  # Tổng hợp doanh số theo tháng

# Tạo biểu đồ đường
plt.figure(figsize=(12, 6))  # Thiết lập kích thước biểu đồ
graph = sns.lineplot(x="Month", y="Total_Sales", hue="Year", data=sales_by_month, marker='o', palette='tab10')  # Tạo biểu đồ đường với các đường màu khác nhau cho từng năm

# Thiết lập các nhãn và dấu tích trên trục y
graph.set_yticks([0, 20000000, 40000000, 60000000, 80000000, 100000000])
graph.set_yticklabels(['0', '20M', '40M', '60M', '80M', '100M'])

# Ghi chú điểm đỉnh (max), đáy (min), điểm bắt đầu (tháng đầu tiên), và điểm kết thúc (tháng cuối cùng)
for year in sales_by_month['Year'].unique():  # Lặp qua từng năm
    data = sales_by_month[sales_by_month['Year'] == year]  # Lọc dữ liệu theo năm
    max_point = data[data['Total_Sales'] == data['Total_Sales'].max()]  # Tìm điểm có doanh số cao nhất
    min_point = data[data['Total_Sales'] == data['Total_Sales'].min()]  # Tìm điểm có doanh số thấp nhất
    start_point = data[data['Month'] == data['Month'].min()]  # Tìm điểm bắt đầu (tháng đầu tiên)
    end_point = data[data['Month'] == data['Month'].max()]  # Tìm điểm kết thúc (tháng cuối cùng)
    
    for _, row in pd.concat([max_point, min_point, start_point, end_point]).iterrows():  # Lặp qua từng điểm cần ghi chú
        graph.text(row['Month'], row['Total_Sales'], f'{int(row["Total_Sales"]):,}', ha='center', va='bottom' if row["Total_Sales"] > 0 else 'top', fontsize=10)  # Ghi chú doanh số tại các điểm này

# Thiết lập nhãn và tiêu đề
graph.set_xlabel("Month")  # Thiết lập nhãn trục x
graph.set_ylabel("Total Sales")  # Thiết lập nhãn trục y
plt.title("Total Sales by Month and Year")  # Thiết lập tiêu đề biểu đồ
plt.show()  # Hiển thị biểu đồ





# UNITS SOLD THEO GIỚI TÍNH
# Thêm cột Gender , sử dụng str.contains để kiểm tra men hay ko , sử dụng boolean để true =  men , false = women
Adidas_data["Gender"] = Adidas_data["Product"].str.contains("Men")
Adidas_data["Gender"] = Adidas_data["Gender"].map({True: "Men", False: "Women"})

# sử dụng map_product_to_category để tham chiếu các sản phẩm lưu vào cột Category
def map_product_to_category(product):
    if 'Apparel' in product:
        return 'Apparel'
    elif 'Footwear' in product:
        if 'Athletic' in product:
            return 'Athletic Footwear'
        else:
            return 'Street Footwear'
    else:
        return 'Other'

Adidas_data['Category'] = Adidas_data['Product'].apply(map_product_to_category)

# Tính tổng số lượng sản phẩm bán được (Units Sold) theo 'Gender' và 'Category'
summary_data = Adidas_data.groupby(['Gender', 'Category'])['Units Sold'].sum().reset_index()

# thiết lập giao diện seaborn
sns.set(style="whitegrid")
# thiết lập kích thước
plt.figure(figsize=(12, 8))
# vẽ biểu đồ bằng seaborn
ax = sns.barplot(data=summary_data, x='Gender', y='Units Sold', hue='Category', palette="pastel", edgecolor='black', ci=None)
# thêm thông số vào cột
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)
# đặt tên 
plt.xlabel('')
plt.ylabel('Units Sold')
plt.title('Units Sold Per Category Per Gender')
plt.legend(title='Category')
# Hiện thị
plt.show()




# TẠO BẢNG TOTAL SALES VỚI GENDER
# Thêm cột 'Gender' vào dữ liệu đã đổi tên
Adidas_data_re["Gender"] = Adidas_data_re["Product"].apply(lambda x: "Male" if "Men" in x else "Female")
# Tính tổng doanh thu theo giới tính
sales_by_gender = Adidas_data_re.groupby('Gender')['Total_Sales'].sum().reset_index()
# Thiết lập kích thước cho biểu đồ
plt.figure(figsize=(8, 5))
# Vẽ biểu đồ tổng doanh thu theo giới tính
figer = sns.barplot(x='Gender', y='Total_Sales', data=sales_by_gender, palette='muted')
figer.set_yticks([0, 100000000, 200000000, 300000000, 400000000, 500000000, 600000000])
figer.set_yticklabels(['0', '100M', '200M', '300M', '400M', '500M', '600M'])
# Thêm nhãn giá trị lên các cột trong biểu đồ
for p in figer.patches:
    figer.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)
# đặt tên 
# Thiết lập tiêu đề và nhãn trục
plt.title('Total Sales by Gender', fontsize=16)
plt.ylabel('Total Sales', fontsize=14)
plt.xlabel('Gender', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
# Hiển thị biểu đồ
plt.tight_layout()
plt.show()



# BIỂU ĐỒ TRÒN
# Chuyển đổi kiểu dữ liệu của cột Total Sales
Adidas_data_re["Total_Sales"] = Adidas_data_re["Total_Sales"].astype(int)
# Lọc dữ liệu theo sản phẩm
str_filtered = Adidas_data_re[Adidas_data_re["Product"].str.contains("Street Footwear")]
apparel_filtered = Adidas_data_re[Adidas_data_re["Product"].str.contains("Apparel")]
athletic_filtered = Adidas_data_re[Adidas_data_re["Product"].str.contains("Athletic Footwear")]
# Tính tổng doanh thu cho từng loại sản phẩm
str_sales = str_filtered["Total_Sales"].sum()
apparel_sales = apparel_filtered["Total_Sales"].sum()
athletic_sales = athletic_filtered["Total_Sales"].sum()
# Vẽ biểu đồ hình tròn
labels_pie = ['Street Footwear', 'Apparel', 'Athletic Footwear']
sales_pie = [str_sales, apparel_sales, athletic_sales]
plt.figure(figsize=(10, 6))
plt.pie(sales_pie, labels=labels_pie, autopct='%1.1f%%', shadow=True)
plt.title('Total Sales Distribution by Product Category')
plt.legend()
plt.axis('equal')
plt.show()






















# Loại bỏ các ký tự đặc biệt
Adidas_data_re['Price_per_Unit'] = Adidas_data_re['Price_per_Unit'].astype(str).str.replace('$', '', regex=True)
Adidas_data_re['Total_Sales'] = Adidas_data_re['Total_Sales'].astype(str).str.replace('[$,]', '', regex=True)
Adidas_data_re['Operating_Profit'] = Adidas_data_re['Operating_Profit'].astype(str).str.replace('[$,]', '', regex=True)
Adidas_data_re['Operating_Margin'] = Adidas_data_re['Operating_Margin'].astype(str).str.replace('%', '', regex=True)


# chuyển đổi các cột dữ liệu thành số
Adidas_data_trans = Adidas_data_re.copy()
Adidas_data_trans['Total_Sales'] = pd.to_numeric(Adidas_data_trans['Total_Sales'])
Adidas_data_trans['Operating_Margin'] = pd.to_numeric(Adidas_data_trans['Operating_Margin'])
Adidas_data_trans['Operating_Profit'] = pd.to_numeric(Adidas_data_trans['Operating_Profit'])
Adidas_data_trans['Units_Sold'] = pd.to_numeric(Adidas_data_trans['Units_Sold'])
Adidas_data_trans['Price_per_Unit'] = pd.to_numeric(Adidas_data_trans['Price_per_Unit'])

# Tách cột Invoice_Date thành nhiều cột riêng biệt (Năm, Tháng, Ngày)
Adidas_data_trans['Invoice_Date'] = pd.to_datetime(Adidas_data_trans['Invoice_Date'], format="%m/%d/%Y")
Adidas_data_trans['Invoice_Year'] = Adidas_data_trans['Invoice_Date'].dt.year
Adidas_data_trans['Invoice_Month'] = Adidas_data_trans['Invoice_Date'].dt.month
Adidas_data_trans['Invoice_Day'] = Adidas_data_trans['Invoice_Date'].dt.day

# Tạo bảng Total sales theo ralaiter
Sales_by_retailers = Adidas_data_trans.groupby('Retailer')['Total_Sales'].sum().reset_index().rename(columns={'Total_Sales': 'Total_Sales_Retail'})
# tạo bảng
plt.figure(figsize=(10, 6))
sns.barplot(data=Sales_by_retailers, x='Retailer', y='Total_Sales_Retail', palette='viridis')
plt.title('Total Sales by Retailer')
plt.ylabel('Total Sales (in millions)')
plt.xlabel('Retailer')
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x * 1e-6)}M"))
plt.show()



# Tóm tắt lợi nhuận theo nhà bán lẻ (Retailer)
Profit_by_retailers = Adidas_data_trans.groupby('Retailer')['Operating_Profit'].sum().reset_index().rename(columns={'Operating_Profit': 'Profit'})
print(Profit_by_retailers.head())

# Tóm tắt doanh thu theo product
Sales_by_Product = Adidas_data_trans.groupby('Product')['Total_Sales'].sum().reset_index().rename(columns={'Total_Sales': 'Product_sales'})
print(Sales_by_Product.head())

#Tóm tắt doanh thu theo phương thức bán hàng sales by method
Method = Adidas_data_trans.groupby('Sales_Method')['Total_Sales'].sum().reset_index().rename(columns={'Total_Sales': 'Value'})
print(Method.head())

# Tính giá trung bình mỗi đơn vị sản phẩm theo từng loại  product
Average_price = Adidas_data_trans.groupby('Product')['Price_per_Unit'].mean().reset_index().rename(columns={'Price_per_Unit': 'Avg_Price_Per_Unit'})
print(Average_price.head())






