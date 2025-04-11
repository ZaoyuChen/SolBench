from utils import load_dataset_from_file_or_folder,truncate_funit,read_jsonl,write_jsonl,ds_error_info_process,ds_retrieve_sc_process
from tqdm import tqdm
import json, os
from openai import OpenAI
from prompt_and_func_repos import modes

def single_mode_sample(cot_mode,ds,ds1):
    # openai batch采样jsonl文件save path
    # 临时文件
    jsonl_save_path='openai_batchinput_temp.jsonl'
    # 参数区end

    selected_mode = modes[cot_mode]
    system_prompt = selected_mode['system_prompt']
    get_user_prompt_1 = selected_mode['get_user_prompt_func_1']
    get_user_prompt_2 = selected_mode['get_user_prompt_func_2']

    requests=[]
    for idx in range(len(ds)):
        d=ds[idx]
        user_prompt_1=get_user_prompt_1(d)
        user_prompt_2=get_user_prompt_2(d)
        assistant_prompt=ds1[idx]['response']['body']['choices'][0]['message']['content']
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_1},
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": user_prompt_2},
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

    write_jsonl(requests,jsonl_save_path)

    client = OpenAI(
        api_key='your api key'
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
    cot_modes=[
        'self_debug',
    ]
    batch_id_dicts=[]
    for length in ['256']:

        # 第一次sample文件
        jsonl_path='sample_results/self_debug_1/256.jsonl'

        ds1=read_jsonl(jsonl_path)

        # 参数区
        clean_fail_ds_path=f"data/fail_ds/context_length_{length}_fail.parquet"
        
        ds=load_dataset_from_file_or_folder(clean_fail_ds_path)
        
        for cot_mode in cot_modes:

            batch_id=single_mode_sample(cot_mode,ds,ds1)
            
            batch_id_dict={
                'length':length,
                'cot_mode':cot_mode,
                'batch_id':batch_id,
            }
            batch_id_dicts.append(batch_id_dict)

            for k,v in batch_id_dict.items():
                print(k, v)
            print("-" * 40)  # 打印分隔线以便更清晰地区分不同消息
        
    # 存储 batch_id_dict json 文件
    batch_id_dict_save_path='batch_id_dicts.jsonl'
    write_jsonl(batch_id_dicts,batch_id_dict_save_path)