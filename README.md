# CS744-Big-Data-System
Spring 2023 Comp Sci 744 Assignments and Projects

## Tasks
Task 1. Write a Scala/Python/Java Spark application that implements the PageRank algorithm.

Task 2. In order to achieve high parallelism, Spark will split the data into smaller chunks called partitions which are distributed across different nodes in the cluster. Partitions can be changed in several ways. For example, any shuffle operation on a DataFrame (e.g., join()) will result in a change in partitions (customizable via user’s configuration). In addition, one can also decide how to partition data when writing DataFrames back to disk. For this task, add appropriate custom DataFrame/RDD partitioning and see what changes.

Task 3. Persist the appropriate DataFrame/RDD(s) as in-memory objects and see what changes.

Task 4. Kill a Worker process and see the changes. You should trigger the failure to a desired worker VM when the application reaches 25% and 75% of its lifetime:

Clear the memory cache using sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches".

Kill the Worker process.
