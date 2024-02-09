#!/bin/sh
  
./spark-3.3.4-bin-hadoop3/bin/spark-submit \
  --master spark://10.10.1.1:7077 \
  task4_wikidata.py "hdfs://10.10.1.1:9000/enwiki-pages-articles/" "hdfs://10.10.1.1:9000/task4_wikidata"
