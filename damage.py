from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col
from pyspark.sql.functions import percentile_approx, mean, max
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

df = df.withColumn("kills", df["kills"].cast(IntegerType()))
df = df.withColumn("damageDealt", df["damageDealt"].cast(FloatType()))

# 过滤出kills为0的行
df_zero_kills = df.filter(df["kills"] == 0)

# 将过滤后的Spark DataFrame转换为Pandas DataFrame
pandas_df_zero_kills = df_zero_kills.select("damageDealt").toPandas()

# 使用Seaborn进行分布图绘制
plt.figure(figsize=(15,10))
plt.title("Damage Dealt by 0 Killers", fontsize=15)
sns.histplot(pandas_df_zero_kills['damageDealt'], kde=False)  # 使用histplot代替distplot，因为distplot在新版Seaborn中已被弃用
plt.xlabel("Damage Dealt")
plt.ylabel("Frequency")
plt.savefig('damage_dealt_by_0_killers.png', format='png', dpi=300)

# 显示图表
plt.show()

# 保存图表为图片

# 停止SparkSession
spark.stop()