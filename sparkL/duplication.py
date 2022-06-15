from pyspark import SparkConf
from pyspark import SparkContext
import os

# 指定JAVA_HOME 地址 (console的时候用）
os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk1.8.0_162'
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3.6'
os.environ['SPARK_HOME'] = '/opt/spark'
os.environ['HADOOP_HOME'] = '/usr/local/hadoop'

# 初始化
conf = SparkConf()
conf.setMaster("local[*]").setAppName("app")

# 实例
sc = SparkContext(conf=conf)

path = 'hdfs://master:9000/First_Test/A.txt'
path_b = 'hdfs://master:9000/First_Test/B.txt'
a = sc.textFile(path)
b = sc.textFile(path_b)

union = a.union(b)
union_dic = union.map(lambda x: x.split("   ")).filter(lambda x:x[0] != '').map(lambda x: (x[0], x[1])).distinct()

union_dic.sortByKey().foreach(print)