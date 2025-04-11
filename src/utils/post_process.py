import os
import argparse
import random
import numpy as np
import re
from tqdm import tqdm
import json
import shutil
from time import time
from copy import deepcopy
import ast
from utils import truncate_funit,load_dataset_from_file_or_folder,set_seed,pre_save_solfile_DISL,brace_truncate ,post_process_and_save_DISL,return_func_body_from_funit, read_jsonl

def brace_truncate(completion):
    """
    stop the completion of func_body at }. func_body 从funit的function的下一行开始
    最后的}比{个数多一个
    返回的位置是}
    """
    open_count = 0
    for idx, char in enumerate(completion):
        if char == '{':
            open_count += 1
        elif char == '}':
            open_count -= 1
            if open_count < 0:
                return completion[:idx+1]  # 返回第一个 } 比 { 多的位置
            
    raise Exception('没有函数体')  # 如果没有找到这样的位置，返回 completion本身

def openai_completion_process_backup(completion):
    # 针对没有生成完，不匹配代码块结束符号
    sol_completion = completion[
        (completion.find('```solidity')+len('```solidity')):
        ]
    try:
        return brace_truncate(completion)
    except Exception as e:
        try:
            return return_func_body_from_funit(sol_completion)
        except Exception as e:
            return sol_completion

def openai_completion_process(completion):
    # 找到所有代码块包裹内容
    pattern = r"```solidity\s*([\s\S]*?)```"
    sol_completions = re.findall(pattern, completion)
    # 尝试返回代码块包裹函数体的情况
    for idx, sol_completion in enumerate(sol_completions, start=1):
        try:        
            return brace_truncate(sol_completion)
        except Exception as e:
            continue
    # 尝试返回代码块包裹完整函数的情况
    for idx, sol_completion in enumerate(sol_completions, start=1):
        try:        
            return return_func_body_from_funit(sol_completion)
        except Exception as e:
            continue

    # 所有情况都不是，可能是没有生成完，没有匹配的代码块
    # 还有情况是，匹配代码块中没有合适的，openai_completion_process_backup会返回第一个代码块
    return openai_completion_process_backup(completion)

def epoch_sample(args,base_save_dir):  
    # 加载openai返回的采样后的文件，相当于text output
    text_outs=read_jsonl(args.openai_outputfile)
    # 对 采样结果排序，按照 custom_id 升序排列
    text_outs = sorted(text_outs, key=lambda x: int(x['custom_id']))
    # 加载测试集文件
    sol_prompt_sc=load_dataset_from_file_or_folder(args.sol_prompt_sc_fname)
    
    for idx,d in enumerate(tqdm(sol_prompt_sc)):
        sol_name=d['sol'].split('/')[-1].split('.')[0]
        funit=d['funit']
        sc_ba=json.loads(d['sc'])
        prompt_i=d['idx']
        source_code=f'{sc_ba[0]}\r\n{funit}\r\n{sc_ba[1]}'
        save_dir=os.path.join(base_save_dir,prompt_i)
        pre_save_solfile_DISL(source_code,save_dir,sol_name)
        
        try:
            assert text_outs[idx]['custom_id'] == str(prompt_i)
        except:
            breakpoint()

        # 取出openai返回的采样后的文件的func_body,对于单采样或多采样都适用
        func_bodys=[]        
        api_type=args.model_type        
        if api_type == 'openai':
            for choice in text_outs[idx]['response']['body']['choices']:
                completion=choice['message']['content']
                func_body=openai_completion_process(completion)
                func_bodys.append(func_body)

        elif api_type == 'claude':
            for choice in text_outs[idx]['result']['message']['content']:
                completion=choice['text']
                func_body=openai_completion_process(completion)
                func_bodys.append(func_body)

        elif api_type == 'deepseek':
            try:
                for choice in text_outs[idx]['choices']:
                    completion=choice['message']['content']
                    func_body=openai_completion_process(completion)
                    func_bodys.append(func_body)
            except:
                for choice in text_outs[idx]['response']['body']['choices']:
                    completion=choice['message']['content']
                    func_body=openai_completion_process(completion)
                    func_bodys.append(func_body)
                
        elif (api_type == 'qwen') or (api_type == 'llama'):
            completion=text_outs[idx]['generated_text'][2]['content']
            func_body=openai_completion_process(completion)
            func_bodys.append(func_body)

        # 将text_out嵌套一层，为后面多采样准备
        post_process_and_save_DISL(args,func_bodys,funit,sc_ba,prompt_i,sol_name,save_dir)


collections=\
[
    {
    'epoch':[0],
    'model':'my_model',
    },
]


if __name__ == '__main__':
    set_seed(seed=0)

    parser = argparse.ArgumentParser(description='输入参数')
    # 路径参数
    parser.add_argument('--base_save_dir', required=True, help='补全的合约函数save dir')
    parser.add_argument('--sol_prompt_sc_fname', required=True, help='采样的文件名')
    parser.add_argument('--openai_outputfile', required=True, help='openai返回的采样后的文件')
    parser.add_argument('--model_type', required=True, help='openai返回的采样后的文件')
    args = parser.parse_args() 
    
    for collection_id,collection in enumerate(collections):
        args.model_name=collection['model']
        args.epoch=collection['epoch']
        for args.ckpt_id in args.epoch:   
            epoch_sample(args,args.base_save_dir)