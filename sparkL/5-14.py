from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
import os
import json

# 指定JAVA_HOME 地址 (console的时候用）
os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk1.8.0_162'
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3.6'
os.environ['SPARK_HOME'] = '/opt/spark'
os.environ['HADOOP_HOME'] = '/usr/local/hadoop'

with open('/home/hadoop/resource/employee.json') as f:
    data = json.load(f)

conf = SparkConf()
conf.setMaster('local[*]').setAppName('app')

sc = SparkContext(conf=conf)

spark = SparkSession.builder.master('local').appName('app').config("spark.some.config.option", "some-value").getOrCreate()

df = spark.read.json('hdfs://master:9000/First_Test/employee.json')
df.show()

df.drop_duplicates().show()

df.drop('id').show()

df.filter(df['age'] > 30).show()

df.groupBy('age').count().show()

df.sort(df['name']).show()

df.head(3)

df.select(df.name.alias('username')).show()

df.agg({'age': 'mean'}).show()

df.agg({'age': 'min'}).show()

text = sc.textFile('hdfs://master:9000/First_Test/employee.txt')
employee = text.map(lambda x: x.split(',')).map(lambda p:Row(id=p[0], name=p[1], age=p[2]))

schemaEmployee = spark.createDataFrame(employee)

employeeRDD = schemaEmployee.rdd.map(lambda p:"id:"+p.id+",name:"+p.name+",age:"+str(p.age))
employeeRDD.foreach(print)

from pyspark.sql.types import *

#设置模式信息
schema = StructType([StructField("id",IntegerType(),True) \
          ,StructField("name",StringType(),True) \
          ,StructField("gender",StringType(),True) \
          ,StructField("age",IntegerType(),True)])

#设置两条数据
employeeRDD = spark.sparkContext.parallelize(["3 Mary F 26","4 Tom M 23"]) \
              .map(lambda x:x.split(" "))

#创建Row对象
rowRDD = employeeRDD.map(lambda p:Row(int(p[0].strip()),p[1].strip() \
         ,p[2].strip(),int(p[3].strip())))

#建立Row对象和模式直接的对应关系
employeeDF = spark.createDataFrame(rowRDD,schema)

#写入数据库
prop={}
prop['user']='root'
prop['password']='123456'
prop['driver']='com.mysql.jdbc.Driver'
employeeDF.write.jdbc("jdbc:mysql://localhost:3306/sparktest",'employee','append',prop)

employeeDF.createOrReplaceTempView("employee")
spark.sql("select max(age),sum(age) from employee").show()

employeeDF.agg({"age":"max"}).show()
employeeDF.agg({"age":"sum"}).show()