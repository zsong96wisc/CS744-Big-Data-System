from pyspark import SparkConf, SparkContext
import sys
import os

conf = SparkConf().setAppName("PageRankWiki_task1")
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

links = filteredLink2EachDest.groupByKey()

ranks = links.map(lambda link: (link[0], 1.0))

for i in range(1, 4):
    contributions = links.join(ranks).flatMap(lambda x: [(destUrl, rank / len(x[1][0])) for destUrl, rank in zip(x[1][0], [x[1][1]] * len(x[1][0]))])

    ranks = contributions.reduceByKey(lambda x, y: x + y).mapValues(lambda sum: 0.15 + (0.85 * sum))

ranks.repartition(1).saveAsTextFile(outputLocation)

sc.stop()