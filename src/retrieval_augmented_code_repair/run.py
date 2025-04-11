from prompt_and_func_repos import modes
from openai import OpenAI
from utils import load_dataset_from_file_or_folder,truncate_funit,write_jsonl
from tqdm import tqdm
import json, os

def single_mode_sample(mode):
    # openai batch采样jsonl文件save path
    # 临时文件
    jsonl_save_path='openai_batchinput_temp.jsonl'
    
    selected_mode = modes[mode]
    system_prompt = selected_mode['system_prompt']
    get_user_prompt = selected_mode['get_user_prompt_func']


    # 生成gpt4omini api batch采样的输入jsonl文件
    requests=[]
    for d in ds:
        user_prompt=get_user_prompt(d)

        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            ]
        
        request={
            "custom_id": d['idx'], 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": "gpt-4o-mini", 
                "messages": messages,
                "temperature":0,
                }}
        requests.append(request)

    # 存储sample jsonl文件
    write_jsonl(requests,jsonl_save_path)     
        
    client = OpenAI(
        api_key='your api_key'
    )

    batch_input_file = client.files.create(
        file=open(jsonl_save_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    client_batches_create=client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "solidity test job"
        }
    )
    return client_batches_create.id

if __name__ == '__main__':
    # 参数区
    # could also be 'ra_self_repair','ra_self_refine',
    cot_modes=[
    'ra_self_edit',
        ]
 
    batch_id_dicts=[] 
    for length in ['256']:    
        fail_ds_path=f"data/fail_ds/context_length_{length}_fail_ra.parquet"
        
        ds=load_dataset_from_file_or_folder(fail_ds_path)

        
        for cot_mode in cot_modes:
            batch_id=single_mode_sample(cot_mode)
            
            batch_id_dict={
                'length':length,
                'cot_mode':cot_mode,
                'batch_id':batch_id,
            }
            batch_id_dicts.append(batch_id_dict)
            
            for k,v in batch_id_dict.items():
                print(k, v)
            print("-" * 40)
            
    # 存储 batch_id_dict json 文件
    batch_id_dict_save_path='batch_id_dicts.jsonl'
    write_jsonl(batch_id_dicts,batch_id_dict_save_path)