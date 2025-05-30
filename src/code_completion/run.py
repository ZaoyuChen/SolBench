from utils import load_dataset_from_file_or_folder,truncate_funit,write_jsonl,read_jsonl
from time import time,sleep
import os,json
from transformers import pipeline
import argparse


system_prompt=\
"""You are a Solidity expert. Given a unfinished Solidity function, your task is to generate a complete implementation of the function. Put your completion within code delimiters:
```solidity
# YOUR CODE HERE
```
For example:
# Given a unfinished Solidity function:
```solidity
    // Set royalty  address
function setRoyaltyAddress(address _address) external onlyOwner {
```

# Your completion is:
```solidity
function setRoyaltyAddress(address _address) external onlyOwner {
    royaltyAddress = _address;
}
```
"""  

# 在query后append task prompt
def get_user_prompt(d):
    instruction_prompt=f"""\
{d['retrieve_sc']}

# Given a unfinished Solidity function:
```solidity
{truncate_funit(d['funit'])}
```

# Your completion is:
"""
    return instruction_prompt


def single_mode_completion(pipe,ds,mode):
    completions_save_path=args.completions_save_path
    
    completions=[]

    cnt=0
    for d in ds:
        
        cnt+=1
        start_time=time()
        user_prompt=get_user_prompt(d)
        
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            completion=model_call(pipe,messages)
            completion['custom_id']=d['idx']
            completions.append(completion)
            
            total_cost_time=time()-total_start_time
            cost_time=time()-start_time
            print(f"mode:{mode} -- ds_idx:{d['idx']} -- time:{cost_time:.2f} -- total time:{total_cost_time:.2f}")
            
        except Exception as e:
            print(e)
            break
        
        if cnt%500==0:
            write_jsonl(completions,completions_save_path)
            print(f"{cnt} Completions saved to {completions_save_path}")
            
    write_jsonl(completions,completions_save_path)
    print(f"Completions saved to {completions_save_path}")

# Use a pipeline as a high-level helper
from transformers import pipeline
def model_call(pipe,messages):

    completions=pipe(messages, max_new_tokens=512)
    # print(completions[0]['generated_text'][2]['content'])
    return completions[0]

if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description='输入参数')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--context_length', required=True)
    parser.add_argument('--completions_save_path', required=True)
    
    args = parser.parse_args() 
    
    total_start_time=time()
    
    pipe = pipeline("text-generation", model=args.model_path)
    mode=args.context_length
    
    ds=load_dataset_from_file_or_folder(f'data/SolBench_length/SolBench_length_{mode}.parquet')
    
    single_mode_completion(pipe,ds,mode)
        