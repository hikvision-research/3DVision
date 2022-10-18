#!/usr/bin/env bash
script_path=$(cd `dirname $0`; pwd)
cd $script_path

python main.py --phase 'train' --name 'demo'


