from pyspark import SparkConf, SparkContext, StorageLevel
import sys

# Initialize SparkContext
conf = SparkConf().setAppName("pagerank_task3_memory")
sc = SparkContext(conf=conf)

num_partitions = 30

# Read input/output path
input_path = sys.argv[1]
output_path = sys.argv[2]

rawData = sc.textFile(input_path).filter(lambda row: not row.startswith("#"))

link2EachDest = rawData.map(lambda row: (row.split("\t")[0], row.split("\t")[1])).persist(storageLevel=StorageLevel(False,True,False,False,1))

links = link2EachDest.groupByKey().partitionBy(num_partitions)

ranks = links.map(lambda link: (link[0], 1.0))

for i in range(1, 11):
    contributions = links.join(ranks).flatMap(lambda x: [(destUrl, x[1][1] / len(x[1][0])) for destUrl in x[1][0]])
    
    ranks = contributions.reduceByKey(lambda x, y: x + y).mapValues(lambda sum: (0.15 + (0.85 * sum)))

ranks.repartition(1).saveAsTextFile(output_path)

sc.stop()