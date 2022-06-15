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

algorithm = sc.textFile('hdfs://master:9000/First_Test/Algorithm.txt')
database = sc.textFile('hdfs://master:9000/First_Test/DataBase.txt')
python = sc.textFile('hdfs://master:9000/First_Test/Python.txt')

courses = algorithm.union(database).union(python)
courses = courses.map(lambda x: x.split(" ")).map(lambda x: (x[0], (int(x[1]), 1)))
average = courses.reduceByKey(lambda x, y: (x[0]+y[0], x[1] + y[1])).map(lambda x: (x[0], round((x[1][0] / x[1][1]), 2)))

courses.collect()
average.foreach(print)