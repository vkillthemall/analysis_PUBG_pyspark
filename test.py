from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import mean, max

# 使用matplotlib和seaborn进行分布图绘制
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import FloatType

# 初始化SparkContext和SparkSession
sc = SparkContext('local', 'spark_project')
sc.setLogLevel('WARN')  # 减少不必要的 LOG 输出
spark = SparkSession(sc)

# 读取CSV文件并创建Dataframe
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .load("hdfs://localhost:9000/data/data_train.csv")
df = df.withColumn("rideDistance", df["rideDistance"].cast(FloatType()))

# 计算rideDistance的平均值、99%分位数和最大值
average_ride_distance = df.select(mean("rideDistance")).collect()[0][0]
ride_distance_99th_percentile = df.selectExpr("percentile_approx(rideDistance, 0.99)").collect()[0][0]

max_ride_distance = df.select(max("rideDistance")).collect()[0][0]

# 打印统计信息
print(f"平均载具行驶距离为 {average_ride_distance:.1f}m, 99% 的玩家行驶了 {ride_distance_99th_percentile}m 或更少, 行驶距离最远的玩家行驶 {max_ride_distance}m.")

# 筛选出rideDistance小于99%分位数的行
df_ride_distance_less_99th = df.filter(df["rideDistance"] < ride_distance_99th_percentile)

# 将筛选后的Spark DataFrame转换为Pandas DataFrame
pandas_df_ride_distance = df_ride_distance_less_99th.select("rideDistance").toPandas()

plt.figure(figsize=(15,10))
plt.title("Ride Distance Distribution", fontsize=15)
sns.histplot(pandas_df_ride_distance['rideDistance'], kde=False)
plt.xlabel("Ride Distance (m)")
plt.ylabel("Frequency")
plt.savefig('ridding_distacne.png', format='png', dpi=300)
plt.show()

# 计算rideDistance为0的人数及其占总人数的百分比
count_zero_ride_distance = df.filter(df["rideDistance"] == 0).count()
total_count = df.count()
percentage_zero_ride_distance = (count_zero_ride_distance / total_count) * 100

print(f"{count_zero_ride_distance} 位玩家 ({percentage_zero_ride_distance:.4f}%) 行驶了 0 米. 他们本局游戏没有使用过载具.")

# 停止SparkSession
spark.stop()