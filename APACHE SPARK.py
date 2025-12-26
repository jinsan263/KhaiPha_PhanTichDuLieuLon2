import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
from pyspark.ml import Pipeline
spark = SparkSession.builder \
    .appName("DuBaoLoiNhuan") \
    .getOrCreate()

# Đọc dữ liệu
df = spark.read.csv('df_cleaned.csv', header= True, inferSchema= True )
df.show(5)

# PHÂN TÍCH MÔ TẢ
# Thông tin chung
df.printSchema()

#BẢNG THỐNG KÊ
# Chọn các cột quan trọng để thống kê
cols = ['Sales', 'Quantity', 'Discount', 'Profit']

# Lấy describe()
desc = df.select(cols).describe()

# Lấy quantile
quantiles = {}
for c in cols:
    q1, med, q3 = df.approxQuantile(c, [0.25, 0.5, 0.75], 0.01)
    quantiles[c] = (q1, med, q3)

# Chuyển quantile thành Spark DataFrame
rows = [
    ("Q1",) + tuple(quantiles[c][0] for c in cols),
    ("Q2",) + tuple(quantiles[c][1] for c in cols),
    ("Q3",) + tuple(quantiles[c][2] for c in cols),
]

quant_df = spark.createDataFrame(rows, ["summary"] + cols)

# Ghép với describe()
final_stats = desc.union(quant_df)

final_stats.show(truncate=False)

#MA TRẬN TƯƠNG QUAN
#Tính ma trận tương quan trong Spark
cols = ["Profit", "Sales", "Quantity", "Discount"]

corr_values = {}
for c1 in cols:
    corr_values[c1] = []
    for c2 in cols:
        corr = df.stat.corr(c1, c2)
        corr_values[c1].append(corr)
# Chuyển sang Pandas để vẽ heatmap
corr_values
corr_df = pd.DataFrame(corr_values, index=cols)
corr_df
# Vẽ biểu đồ Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Biểu đồ tương quan')
plt.show()

#PHÂN TÍCH DỰ BÁO
# Loại bỏ các biến không sử dụng
cols_remove = ['Year', 'TE_Ship_Mode', 'TE_Segment', 'TE_Region', 'Shipping Days']
df = df.drop(*cols_remove)

#Gom các biến độc lập thành vector đặc trưng
assembler = VectorAssembler(
    inputCols=["Sales", "Quantity", "Discount"],
    outputCol="features_raw"
)
df_vec = assembler.transform(df)

# Chuẩn hóa
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True
)

# Khởi tạo mô hình Hồi quy tuyến tính
lr = LinearRegression(
    featuresCol="features",
    labelCol="Profit"
)

#Tạo pipeline
pipeline_lr = Pipeline(stages=[assembler, scaler, lr])

# Chia dữ liệu
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
# Huấn luyện mô hình
model_lr = pipeline_lr.fit(train_df)
# Dự báo
pred_lr = model_lr.transform(test_df)

# Đánh giá mô hình
evaluator_rmse = RegressionEvaluator(
    labelCol="Profit",
    predictionCol="prediction",
    metricName="rmse"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="Profit",
    predictionCol="prediction",
    metricName="r2"
)
rmse = evaluator_rmse.evaluate(pred_lr)
mse = rmse ** 2
r2 = evaluator_r2.evaluate(pred_lr)

metrics_df = spark.createDataFrame([
    Row(Metric="MSE",  Value=mse),
    Row(Metric="RMSE", Value=rmse),
    Row(Metric="R2",   Value=r2)
])
metrics_df.show()


# Chỉ lấy tập test của 1 lần chạy
preds_pd = pred_lr.select("Profit", "prediction").toPandas()

y_test = preds_pd["Profit"]
y_pred = preds_pd["prediction"]

# Vẽ biểu đồ giá trị dự báo và giá trị thực tế
plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Lợi nhuận thực tế")
plt.ylabel("Lợi nhuận dự báo")
plt.title("So sánh lợi nhuận thực tế và lợi nhuận dự báo\n(Mô hình hồi quy tuyến tính)")
plt.grid(True)

plt.show()


