#!/bin/bash
## switch to script's absolute path, then you can use relative path no limit
script_path=$(cd `dirname $0`; pwd)
cd $script_path

python train.py --name "DeepFit_ours"  --model_name "DeepFit_HG"  --batchSize 256 --points_per_patch 256

# python train.py --name "AdaFit_ours"  --model_name "AdaFit_ms_HG"  --batchSize 256 --points_per_patch 700

# python train.py --name "GraphFit_ours"  --model_name "GraphFit_ms_HG"  --batchSize 256


