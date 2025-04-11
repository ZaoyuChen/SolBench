# count passed models and paired prompts for each file. 
import os, json
import pandas as pd
from collections import OrderedDict
from utils import cal_passrate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='输入参数')
    parser.add_argument('--base_path', required=True)    
    args = parser.parse_args()

    base_path=args.base_path
    pass_rate_dict=cal_passrate(base_path)
