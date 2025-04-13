from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import IntegerType,FloatType

# 初始化SparkContext和SparkSession
sc = SparkContext('local', 'spark_project')
sc.setLogLevel('WARN')  # 减少不必要的 LOG 输出
spark = SparkSession(sc)

# 读取CSV文件并创建Dataframe
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .load("hdfs://localhost:9000/data/data_train.csv")

df = df.withColumn("numGroups", df["numGroups"].cast(IntegerType()))
df = df.withColumn("kills", df["kills"].cast(IntegerType()))
df = df.withColumn("winPlacePerc", df["winPlacePerc"].cast(FloatType()))

# 筛选出单人、双人和组队游戏
solos = df.filter(col("numGroups") > 50)
duos = df.filter((col("numGroups") > 25) & (col("numGroups") <= 50))
squads = df.filter(col("numGroups") <= 25)

# 计算每种游戏模式的数量及其百分比
total_games = df.count()
solo_count = solos.count()
duo_count = duos.count()
squad_count = squads.count()


solo_percentage = (solo_count / total_games) * 100
duo_percentage = (duo_count / total_games) * 100
squad_percentage = (squad_count / total_games) * 100

# 打印游戏模式的数量及其百分比（中文输出）
print(f"共有 {solo_count} 场（{solo_percentage:.2f}%）单人游戏，{duo_count} 场（{duo_percentage:.2f}%）双人游戏，以及 {squad_count} 场（{squad_percentage:.2f}%）组队游戏。")

# 将筛选后的Spark DataFrame转换为Pandas DataFrame以进行绘图
pandas_solos = solos.select("kills", "winPlacePerc").toPandas()
pandas_duos = duos.select("kills", "winPlacePerc").toPandas()
pandas_squads = squads.select("kills", "winPlacePerc").toPandas()

# 使用matplotlib和seaborn进行点图绘制
f, ax1 = plt.subplots(figsize=(20,10))
sns.pointplot(x='kills', y='winPlacePerc', data=pandas_solos, color='black')
sns.pointplot(x='kills', y='winPlacePerc', data=pandas_duos, color='#CC0000')
sns.pointplot(x='kills', y='winPlacePerc', data=pandas_squads, color='#3399FF')
plt.text(37, 0.6, 'Solos', color='black', fontsize=17, style='italic')
plt.text(37, 0.55, 'Duos', color='#CC0000', fontsize=17, style='italic')
plt.text(37, 0.5, 'Squads', color='#3399FF', fontsize=17, style='italic')
plt.xlabel('Number of kills', fontsize=15, color='blue')
plt.ylabel('Win Percentage', fontsize=15, color='blue')
plt.title('Solo vs Duo vs Squad Kills', fontsize=20, color='blue')
plt.grid()

plt.savefig('mode_with_win_rate.png', format='png', dpi=300)
plt.show()

# 停止SparkSession
spark.stop()