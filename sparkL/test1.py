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

path = 'hdfs://master:9000/First_Test/chapter4-data01.txt'
dataset = sc.textFile(path)

# 学生总数
res = dataset.map(lambda x: x.split(',')).map(lambda x: x[0])
distinct = res.distinct()

# 开设课程
class_num = dataset.map(lambda x: x.split(',')).map(lambda x: x[1]).distinct()

# Tom's average score
tom = dataset.map(lambda x : x.split(',')).filter(lambda x: x[0] == 'Tom').map(lambda x: float(x[2]))
# tom.mean()

# The number of courses which student chosen
course = dataset.map(lambda x: x.split(',')).groupBy(lambda x:x[0])
test = dataset.map(lambda x : x.split(',')).map(lambda x: (x[0], (x[1], x[2])))
# test.groupByKey().mapValues(len).collect()
# test.groupByKey().mapValues(list).collect()


# Database
database = dataset.map(lambda x: x.split(',')).filter(lambda x: x[1] == 'DataBase')
# database.collect()


# average
average = dataset.map(lambda x: x.split(',')).map(lambda x: (x[1], (int(x[2]), 1)))
average.collect()
tmp_reduce = average.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

tmp_reduce.collect()

result = tmp_reduce.map(lambda x: (x[0], (round((x[1][0] / x[1][1]), 2))))


# accumulator
accumulator = sc.accumulator(0)
database.foreach(lambda x: accumulator.add(1))


print("Student's number: \t\t", distinct.count())
print("Course's number: \t\t", class_num.count())
print("Tom's average score: \t\t", tom.mean())
print("The number of courses which student chosen: \t\n")
test.groupByKey().mapValues(len).collect()
print("The number of DataBase course which be chosen: \t\t", database.count())
print("Each course's average score: \t\n")
result.foreach(print)
print("The number of DataBase course which be chosen(Use accumulator): \t\t", accumulator.value)

