#!/bin/bash

python main.py --master-ip 172.18.0.2 --num-nodes 4 --rank 1
python main.py --master-ip 172.18.0.2 --num-nodes 4 --rank 2
python main.py --master-ip 172.18.0.2 --num-nodes 4 --rank 3
