from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
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

print(df.schema)
df.show(5)
df = df.withColumn("kills", df["kills"].cast(IntegerType()))
df = df.withColumn("damageDealt", df["damageDealt"].cast(FloatType()))

# 统计平均击杀
average_kills = df.select(mean("kills")).collect()[0][0]
# 统计最多击杀
max_kills = df.select(max("kills")).collect()[0][0]
# 统计99%玩家的击杀数
kills_99th_percentile = df.selectExpr("percentile_approx(kills, 0.99)").collect()[0][0]

print(f"平均击杀为 {average_kills:.4f} , 99% 的玩家击杀了 {kills_99th_percentile} 或以下, 最高的击杀记录为 {max_kills}.")

# 将Spark DataFrame转换为Pandas DataFrame
pandas_df = df.select("kills").toPandas()

# 使用apply方法替换超过99%分位数的击杀数为'8+'
pandas_df['kills'] = pandas_df['kills'].apply(lambda x: '8+' if x > kills_99th_percentile else x)

# 使用Seaborn进行绘图
# 由于kills列现在包含字符串，我们需要将其转换为字符串类型以确保正确绘图
sns.countplot(x=pandas_df['kills'].astype(str))
plt.title("Kill Count", fontsize=15)
plt.xticks(rotation=45)

plt.savefig('kill_count_plot.png', format='png', dpi=300)



# 停止SparkSession
spark.stop()