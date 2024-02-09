import os
from pyspark import SparkConf, SparkContext
import sys

num_partitions = 30

def sampling_based_partition(data, num_partitions):
    data_list = data.collect()

    partition_size = len(data_list) // num_partitions

    np.random.shuffle(data_list)
    partitions = [data_list[i*partition_size:(i+1)*partition_size] for i in range(num_partitions)]

    remainder = len(data_list) % num_partitions
    for i in range(remainder):
        partitions[i].append(data_list[num_partitions*partition_size + i])

    return partitions

conf = SparkConf().setAppName("PageRankWiki_task2")
sc = SparkContext(conf=conf)

inputFiles = sys.argv[1]
outputLocation = sys.argv[2]
input_links = []

def parse_neighbors(line, flag):
    return line.split("\t")

for input_file in os.listdir("/proj/uwmadison744-s24-PG0/data-part3/enwiki-pages-articles/"):
    if input_file[0] != "." and input_file.startswith("link"):
        input_inputFiles = os.path.join(inputFiles, input_file)
        input_lines = sc.textFile(input_inputFiles)
        # input_link = input_lines.map(
        #     lambda line: parse_neighbors(line, flag=False)
        # )
        input_links.append(input_lines)

# for input_file in os.listdir(inputFiles):
#     if input_file[0] != "." and input_file.startswith("link"):
#         input_inputFiles = os.path.join(inputFiles, input_file)
#         input_lines = sc.textFile(input_inputFiles)
#         input_link = input_lines.map(
#             lambda line: parse_neighbors(line, flag=False)
#         )
#         input_links.append(input_link)

rawDataRdd = sc.union(input_links)

lowerCasedData = rawDataRdd.map(lambda row: row.lower())

filteredData = lowerCasedData.filter(lambda row: len(row.split("\t")) == 2)

link2EachDest = filteredData.map(lambda row: (row.split("\t")[0], row.split("\t")[1]))

filteredLink2EachDest = link2EachDest.filter(lambda row: ":" not in row[1] or row[1].startswith("Category:"))

# partitions = sampling_based_partition(filteredLink2EachDest, num_partitions)

# partition_rdds = [sc.parallelize(partition) for partition in partitions]

# links = partition_rdds[0].groupByKey()

links = filteredLink2EachDest.groupByKey().partitionBy(num_partitions)

ranks = links.map(lambda link: (link[0], 1.0)).partitionBy(num_partitions)

for i in range(1, 11):
    contributions = links.join(ranks).flatMap(lambda x: [(destUrl, rank / len(x[1][0])) for destUrl, rank in zip(x[1][0], [x[1][1]] * len(x[1][0]))])
    
    ranks = contributions.reduceByKey(lambda x, y: x + y).mapValues(lambda sum: 0.15 + (0.85 * sum)).partitionBy(num_partitions)

ranks.repartition(1).saveAsTextFile(outputLocation)

sc.stop()
