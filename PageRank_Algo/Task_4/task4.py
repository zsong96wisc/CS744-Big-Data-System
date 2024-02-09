from pyspark import SparkConf, SparkContext, StorageLevel
import sys, subprocess

# Initialize SparkContext
conf = SparkConf().setAppName("pagerank_task4")
sc = SparkContext(conf=conf)

num_partitions = 30

# Read input/output path
input_path = sys.argv[1]
output_path = sys.argv[2]

rawData = sc.textFile(input_path).filter(lambda row: not row.startswith("#"))

link2EachDest = rawData.map(lambda row: (row.split("\t")[0], row.split("\t")[1])).persist(storageLevel=StorageLevel(False,True,False,False,1))

links = link2EachDest.groupByKey().partitionBy(num_partitions)

ranks = links.map(lambda link: (link[0], 1.0))

num_iterations = 10

for i in range(1, 11):
    contributions = links.join(ranks).flatMap(lambda x: [(destUrl, x[1][1] / len(x[1][0])) for destUrl in x[1][0]])
    
    ranks = contributions.reduceByKey(lambda x, y: x + y).mapValues(lambda sum: (0.15 + (0.85 * sum)))

    if i == int(0.25 * num_iterations or i == num_iterations):
        subprocess.call(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_cache"])
        print(f"Killing a worker process at {i} iterations")
        subprocess.call(["ssh", "zsong96@worker-20240207182902-10.10.1.1-46307", "sudo", "pkill", "-9", "java"])

ranks.repartition(1).saveAsTextFile(output_path)

sc.stop()