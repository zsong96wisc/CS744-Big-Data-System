#!/bin/sh

./spark-3.3.4-bin-hadoop3/bin/spark-submit \
  --master spark://10.10.1.1:7077 \
  task1.py "hdfs://10.10.1.1:9000/web-BerkStan.txt" "hdfs://10.10.1.1:9000/task1_output7"