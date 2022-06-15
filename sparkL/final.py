import numpy as np
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import DateType
import os
import pandas as pd

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
spark = SparkSession.builder.master("local[*]").appName("app").getOrCreate()

path = ['hdfs://master:9000/Covid-19/covid-19-{}.csv'.format(i) for i in range(1, 21)]
data = spark.read.option("header", True).csv(path)
data = data.withColumn('case_month', data['case_month'].cast(DateType()))

def plot_map():

    after_filter = data.filter("case_month >= '2022-01-01'")
    after_filter = after_filter.select(['res_state'])
    after_filter = after_filter.groupBy('res_state').count()
    after_filter.coalesce(1).write.mode('overwrite').options(header=True).csv('hdfs://master:9000/Covid-19-result/map1')
    print('Successful')

def plot_pie():
    group = data.select('age_group')
    group = group.filter('age_group != "null" and age_group != "Missing"')
    group = group.groupBy('age_group').count()
    group.coalesce(1).write.mode('overwrite').options(header=True).csv('hdfs://master:9000/Covid-19-result/pie1')
    print('Successful')

def plot_bar():
    date = data.select(['case_month', 'death_yn']).filter('death_yn == "Yes"').dropna()
    date = date.groupBy('case_month').count()
    date.coalesce(1).write.mode('overwrite').options(header=True).csv('hdfs://master:9000/Covid-19-result/death_case')
    print('Successful')

def confirmed():
    confirm = data.select(['case_month', 'death_yn']).filter("death_yn == 'Yes' or death_yn == 'No'")
    confirm = confirm.groupBy('case_month').count()
    confirm.coalesce(1).write.mode('overwrite').options(header=True).csv('hdfs://master:9000/Covid-19-result/confirmed1')
    print('Successful')

def death_rate():
    confirm = data.select(['case_month', 'death_yn']).filter("death_yn == 'Yes' or death_yn == 'No'")
    confirm = confirm.groupBy('case_month').count()
    death = data.select(['case_month', 'death_yn']).filter("death_yn == 'Yes'")
    death = death.groupBy('case_month').count()
    death = death.toPandas()
    confirm = confirm.toPandas()
    death.sort_values(['case_month'], inplace=True)
    confirm.sort_values(['case_month'], inplace=True)

    death = pd.Series(death['count'].to_list(), index=death['case_month'].to_list())
    confirm = pd.Series(confirm['count'].to_list(), index=confirm['case_month'].to_list())
    rate = (death.astype(float) / confirm.astype(float)) * 100
    rate = rate.reset_index()
    rate = rate.rename(columns={'index': 'case_month', 0: 'rate'})
    rate = spark.createDataFrame(rate)
    rate.coalesce(1).write.mode('overwrite').options(header=True).csv('hdfs://master:9000/Covid-19-result/rate1')
    print('Successful')


def asy_rate():
    status = data.select(['case_month', 'symptom_status']).filter(
        "symptom_status != 'Missing' and symptom_status != 'Unknown'")
    asymptomatic = data.select(['case_month', 'symptom_status']).filter("symptom_status == 'Asymptomatic'")
    asymptomatic.show()

    status = status.groupBy('case_month').count()
    asy = asymptomatic.groupBy('case_month').count()

    asy = asy.toPandas().sort_values(['case_month'])
    status = status.toPandas().sort_values(['case_month'])

    asy.dropna(axis=0, inplace=True)
    status.dropna(axis=0, inplace=True)

    asy = pd.Series(asy['count'].to_list(), index=asy['case_month'].to_list())
    status = pd.Series(status['count'].to_list(), index=status['case_month'].to_list())

    rate = (asy / status) * 100

    rate = rate.reset_index()
    rate = rate.rename(columns={'index': 'case_month', 0: 'rate'})
    rate = spark.createDataFrame(rate)
    rate.coalesce(1).write.mode('overwrite').options(header=True).csv('hdfs://master:9000/Covid-19-result/asy_rate1')
    print('Successful')


plot_map()
plot_bar()
plot_pie()
asy_rate()
confirmed()
death_rate()


