"""
将采样错误的数据集拿出来，append错误原因和第一次生成funit，再append根据错误检索的上下文
"""
if __name__ == '__main__':
    length='256'
    # clean数据集
    file_folder_path='data/SolBench.parquet'
    # fail采样dir
    base_path='diffusc/sample_results/context_length_256'
    # fail数据集save path
    clean_fail_save_path=f'data/fail_ds/context_length_256_fail_ra.parquet'
    
    retrieve_num=2
    retrieve_len=1

    from utils import extract_prompt_i_from_error_infos_in_sample_dir
    # 根据错误信息原因提取prompt_i
    extract_idxs=extract_prompt_i_from_error_infos_in_sample_dir(base_path,r'my_model_epoch0_prompt(\d+)_sample0')

    # 读取ds数据集
    from utils import return_prompt_i_from_error_info,load_dataset_from_file_or_folder,save_list_of_dir_as_parquet

    ds=load_dataset_from_file_or_folder(file_folder_path)

    # 根据extract_idxs提取测试集
    ds_fail=[]
    for d in ds:
        if d['idx'] in extract_idxs:
            ds_fail.append(d)

    # 将错误原因append到数据集里
    from utils import return_prompt_i_from_error_info,load_dataset_from_file_or_folder,save_list_of_dir_as_parquet,read_error_infos_from_sample_dir

    prompt_i_error_infos=read_error_infos_from_sample_dir(base_path,r'my_model_epoch0_prompt(\d+)_sample0')

    # 将第一次生成的funit提取出来并入文件
    from utils import read_funit_gen_from_sample_dir

    # 调用函数处理文件
    prompt_i_funitgen = read_funit_gen_from_sample_dir(base_path,r'my_model_epoch0_prompt(\d+).txt')

    import json
    ds_fail_append_fe=[]
    for d in ds_fail:
        d['funitgen']=prompt_i_funitgen[d['idx']]
        d['error_infos']=json.dumps(prompt_i_error_infos[d['idx']])
        ds_fail_append_fe.append(d)
        
    # 将检索信息append到错误数据集
    from parser_retrieve_for_error_info import main
    from solidity_parser import parser
    import json,re
    import logging
    from tqdm import tqdm

    ds_fail_append_fer=[]
    for d in tqdm(ds_fail_append_fe):
        error_infos=json.loads(d['error_infos'])
        sc='\r\n\r\n'.join(json.loads(d['sc_ba']))
        retrieve_sc=main(d['funit'],d['funitgen'],sc,error_infos,retrieve_num,retrieve_len)
        d['retrieve_sc']=json.dumps(retrieve_sc)
        ds_fail_append_fer.append(d)

    save_list_of_dir_as_parquet(ds_fail_append_fer,clean_fail_save_path)