from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col,mean,max
from pyspark.sql.types import IntegerType, FloatType

import matplotlib.pyplot as plt
import seaborn as sns

# 初始化SparkContext和SparkSession
sc = SparkContext('local', 'spark_project')
sc.setLogLevel('WARN')  # 减少不必要的 LOG 输出
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

# 读取CSV文件并创建Dataframe
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .load("hdfs://localhost:9000/data/data_train.csv")


df = df.withColumn("walkDistance", df["walkDistance"].cast(FloatType()))  # 确保walkDistance列是浮点数类型
df = df.withColumn("rideDistance", df["rideDistance"].cast(FloatType()))
# 计算walkDistance的99%分位数
walkDistance_99th_percentile = df.selectExpr("percentile_approx(walkDistance, 0.99)").collect()[0][0]

# 筛选出walkDistance小于99%分位数的行
df_walkDistance_less_99th = df.filter(col("walkDistance") < walkDistance_99th_percentile)

# 将筛选后的Spark DataFrame转换为Pandas DataFrame
pandas_df_walkDistance = df_walkDistance_less_99th.select("walkDistance").toPandas()


# # 使用matplotlib和seaborn进行分布图绘制
# plt.figure(figsize=(15,10))
# plt.title("Walking Distance Distribution", fontsize=15)
# sns.histplot(pandas_df_walkDistance['walkDistance'], kde=False)  # 使用histplot代替distplot，因为distplot在新版Seaborn中已被弃用
# plt.xlabel("Walking Distance")
# plt.ylabel("Frequency")
#
# plt.savefig('walking_distance_distribution.png', format='png', dpi=300)
#
# # 显示图表
# plt.show()

# ride
# 计算rideDistance的平均值、99%分位数和最大值
average_ride_distance = df.select(mean("rideDistance")).collect()[0][0]
ride_distance_99th_percentile = df.selectExpr("percentile_approx(rideDistance, 0.99)").collect()[0][0]
max_ride_distance = df.select(max("rideDistance")).collect()[0][0]

# 打印统计信息
print(f"The average person drives for {average_ride_distance:.1f}m, 99% of people have driven {ride_distance_99th_percentile}m or less, while the formula 1 champion drove for {max_ride_distance}m.")

# 筛选出rideDistance小于99%分位数的行
df_ride_distance_less_99th = df.filter(df["rideDistance"] < ride_distance_99th_percentile)

# 将筛选后的Spark DataFrame转换为Pandas DataFrame
pandas_df_ride_distance = df_ride_distance_less_99th.select("rideDistance").toPandas()

plt.figure(figsize=(15,10))
plt.title("Ride Distance Distribution", fontsize=15)
sns.histplot(pandas_df_ride_distance['rideDistance'], kde=False)
plt.xlabel("Ride Distance (m)")
plt.ylabel("Frequency")
plt.savefig('riding_distance_distribution.png', format='png', dpi=300)
plt.show()

# 计算rideDistance为0的人数及其占总人数的百分比
count_zero_ride_distance = df.filter(df["rideDistance"] == 0).count()
total_count = df.count()
percentage_zero_ride_distance = (count_zero_ride_distance / total_count) * 100

print(f"{count_zero_ride_distance} players ({percentage_zero_ride_distance:.4f}%) drove for 0 meters. This means that they don't have a driving license yet.")

# 停止SparkSession
spark.stop()