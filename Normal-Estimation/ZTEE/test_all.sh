
#!/bin/bash
## switch to script's absolute path, then you can use relative path no limit
script_path=$(cd `dirname $0`; pwd)
cd $script_path

Model=DeepFit_ours

python test.py --script_path "./log/${Model}" --testset 'testset_all.txt'  --start_epoch 700

python evaluate.py --normal_results_path "./log/${Model}"

