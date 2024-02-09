from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import sys

def skip_line(x):
    if not "#" in x:
        return x

def split_line(x):
    return x.split("\t")

def swap_line(x):
    return [x[1], x[0]]

def cal_contr(x):
    res = []
    for i in x[1][1]:
        res.append((i, x[1][0] / len(x[1][1])))
    return res

# Initialize SparkSession
# spark = SparkSession.builder \
#     .appName("pagerank") \
#     .config("spark.driver.memory", "30g") \
#     .config("spark.executor.memory", "30g") \
#     .config("spark.executor.cores", "5") \
#     .config("spark.executor.cpus", "1") \
#     .getOrCreate()

# Create a SparkConf object to configure the application
conf = SparkConf().setAppName("pagerank_task1")
sc = SparkContext(conf=conf)

# Read input/output path
input_path = sys.argv[1]
output_path = sys.argv[2]

# Read Edgelist file
edgelist = sc.textFile(input_path).filter(skip_line)
edgelist = edgelist.map(split_line)
# edgelist = edgelist.map(swap_line)
# print(edgelist.take(10))

# Create adjacency list
adjacency_list = edgelist.groupByKey()
# print(adjacency_list.mapValues(list).collect()[:10])

# Create rank list
rank_list = adjacency_list.mapValues(lambda x: 1.0)
# print(rank_list.collect()[:10])

# PageRank Algo
for _ in range(1, 11):
    contr_list = rank_list.join(adjacency_list).flatMap(cal_contr)
    rank_list = contr_list.reduceByKey(lambda x1, x2: x1 + x2).mapValues(lambda x: 0.15 + 0.85 * x)

# print(rank_list.collect()[:10])

rank_list.repartition(1).saveAsTextFile(output_path)

sc.stop()