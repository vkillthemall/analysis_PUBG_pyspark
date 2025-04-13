from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import mean, max

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import IntegerType, FloatType

# 初始化SparkContext和SparkSession
sc = SparkContext('local', 'spark_project')
sc.setLogLevel('WARN')  # 减少不必要的 LOG 输出
spark = SparkSession(sc)

# 读取CSV文件并创建Dataframe
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .load("hdfs://localhost:9000/data/data_train.csv")

df = df.withColumn("heals", df["heals"].cast(IntegerType()))
df = df.withColumn("boosts", df["boosts"].cast(IntegerType()))
df = df.withColumn("winPlacePerc", df["winPlacePerc"].cast(FloatType()))

# 计算heals和boosts的平均值、99%分位数和最大值
average_heals = df.select(mean("heals")).collect()[0][0]
heals_99th_percentile = df.selectExpr("percentile_approx(heals, 0.99)").collect()[0][0]
max_heals = df.select(max("heals")).collect()[0][0]

average_boosts = df.select(mean("boosts")).collect()[0][0]
boosts_99th_percentile = df.selectExpr("percentile_approx(boosts, 0.99)").collect()[0][0]

max_boosts = df.select(max("boosts")).collect()[0][0]

# 打印统计信息
print(f"平均每个人使用 {average_heals:.1f} 个治疗物品，99%的人使用 {heals_99th_percentile} 或更少，而最多的人使用了 {max_heals} 个。")
print(f"平均每个人使用 {average_boosts:.1f} 个增益物品，99%的人使用 {boosts_99th_percentile} 或更少，而最多的人使用了 {max_boosts} 个。")


# 筛选出heals和boosts小于99%分位数的行
df_filtered = df.filter((df["heals"] < heals_99th_percentile) & (df["boosts"] < boosts_99th_percentile))

# 将筛选后的Spark DataFrame转换为Pandas DataFrame
pandas_df_filtered = df_filtered.select("heals", "boosts", "winPlacePerc").toPandas()

# 使用matplotlib和seaborn进行点图绘制
f, ax1 = plt.subplots(figsize=(20,10))
sns.pointplot(x='heals', y='winPlacePerc', data=pandas_df_filtered, color='lime')
sns.pointplot(x='boosts', y='winPlacePerc', data=pandas_df_filtered, color='blue')
plt.text(4, 0.6, 'Heals', color='lime', fontsize=17, style='italic')
plt.text(4, 0.55, 'Boosts', color='blue', fontsize=17, style='italic')
plt.xlabel('Number of heal/boost items', fontsize=15, color='blue')
plt.ylabel('Win Percentage', fontsize=15, color='blue')
plt.title('Heals vs Boosts', fontsize=20, color='blue')
plt.grid()
plt.savefig('heal.png', format='png', dpi=300)

plt.show()

# 停止SparkSession
spark.stop()