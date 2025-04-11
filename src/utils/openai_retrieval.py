from utils import read_jsonl
import argparse

import json
from openai import OpenAI
from utils import write_jsonl

def check_batch_id_dicts(batch_id_dicts):
    batch_id_dicts_w_file_id=[]
    for batch_id_dict in batch_id_dicts:
        [print(k, v) for k,v in batch_id_dict.items()]
        client_batches_retrieve=client.batches.retrieve(batch_id_dict['batch_id'])
        print(client_batches_retrieve.status,client_batches_retrieve.output_file_id,client_batches_retrieve.request_counts)
        print('-'*40)
        
        if client_batches_retrieve.output_file_id is not None:
            batch_id_dict['file_id']=client_batches_retrieve.output_file_id
            batch_id_dicts_w_file_id.append(batch_id_dict)
    
    return batch_id_dicts_w_file_id

def save_sample_from_file_id_dict(batch_id_dicts_w_file_id):
    for batch_id_dict_w_file_id in batch_id_dicts_w_file_id:
        print('取回结果ing')
        file_response = client.files.content(batch_id_dict_w_file_id['file_id'])
        responses=[]
        for line in file_response.text.splitlines():
            response=json.loads(line)
            responses.append(response)
    
        output_jsonl_save_path=f'sample_results/{args.save_dirname}/context_length_{batch_id_dict_w_file_id['length']}.jsonl'
        
        write_jsonl(responses,output_jsonl_save_path)
        print(f'储存到: {output_jsonl_save_path}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='输入参数')
    parser.add_argument('--save_dirname', required=True)
    args = parser.parse_args() 
    
    client = OpenAI(api_key='your api key')

    batch_id_dict_path='batch_id_dicts.jsonl'
    batch_id_dicts=read_jsonl(batch_id_dict_path)

    batch_id_dicts_w_file_id=check_batch_id_dicts(batch_id_dicts)

    save_sample_from_file_id_dict(batch_id_dicts_w_file_id)