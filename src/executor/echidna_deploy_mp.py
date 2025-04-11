import subprocess
import json
import os
import shutil
from collections import defaultdict
from time import time,sleep
from multiprocessing import Process,active_children
import argparse

def diffusc(command):
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100,encoding='utf-8')
    if ret.returncode == 0:
        args.sample_result['compile'].append(args.sample_name)
        return True
    else:
        args.sample_result['compile_fail'].append(args.sample_name)
        args.error_info[args.sample_name]=ret.stdout+'\n'+ret.stderr
        return False

def echidna(command, max_retries=100000):
    ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=100, encoding='utf-8')
    if ret.returncode == 0:
        args.sample_result['pass'].append(args.sample_name)
    elif 'echidna: ABI is empty, are you sure your constructor is right?' in ret.stderr:
        args.sample_result['pass'].append(args.sample_name)
    else:
        args.sample_result['pass_fail'].append(args.sample_name)
        args.error_info[args.sample_name] = ret.stdout + '\n' + ret.stderr

def single_process(sol_nums_batch,process_id):
    process_start_time=time()
    for id_sol_num,sol_num in enumerate(sol_nums_batch,start=1):
        args.sample_result=defaultdict(list)
        args.error_info={}
        origin_name=os.listdir(os.path.join(base_path,sol_num,'origin'))[0]
        origin_file=os.path.join(base_path,sol_num,'origin',origin_name)
        for id_sample_sol,sample_sol in enumerate(os.listdir(os.path.join(base_path,sol_num,'test'))):
            args.sample_name=sample_sol
            print(f'Process {process_id}--base_paths--{base_path_idx}/{len(base_paths)}--sol_num--{id_sol_num}/{len(sol_nums_batch)}--sample_sol--{id_sample_sol}/{len(os.listdir(os.path.join(base_path,sol_num,"test")))}--cost_time--{(time()-process_start_time):.2f}\n')
            for fname in os.listdir(os.path.join(base_path,sol_num,'test',sample_sol)):
                if fname.endswith('sol'):
                    sample_file=os.path.join(base_path,sol_num,'test',sample_sol,fname)
                    diffu_out_dir=os.path.join(base_path,sol_num,'test',sample_sol,'diffusc_out')
                    compile_result=diffusc(f'diffusc {origin_file} {sample_file} -L 100 -d {diffu_out_dir}')
                    if compile_result:
                        echidna(f'echidna {os.path.join(diffu_out_dir,"DiffFuzzUpgrades.sol")} --contract DiffFuzzUpgrades --config {os.path.join(diffu_out_dir,"CryticConfig.yaml")}')
        with open(os.path.join(base_path,sol_num,'diffusc_echidna_result.json'),'w') as f:
            json.dump(args.sample_result,f,indent=4)
        with open(os.path.join(base_path,sol_num,'diffusc_echidna_error_info.json'),'w') as f:
            json.dump(args.error_info,f,indent=4)            
    print(f'Process {process_id} Done! Total time {(time()-start_time):.2f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='输入参数')
    parser.add_argument('--base_path', required=True)    
    args = parser.parse_args()
    
    base_path=args.base_path
    process_list=[]
    max_processes=144
    start_time=time()

    sol_nums=os.listdir(base_path)
            
    for idx,sol_num in enumerate(sol_nums,start=1):
        # 跳过已检测
        result_path=os.path.join(base_path,sol_num,'diffusc_echidna_result.json')
        if os.path.exists(result_path):
            continue
        else:
            # 跳过已检测end
            batch=[sol_num]
            while len(active_children()) >= max_processes:
                # 等待任意一个子进程结束
                finished = False
                for p in process_list:
                    if not p.is_alive():
                        p.join()  # 清理已经结束的进程
                        process_list.remove(p)
                        finished = True
                        break
                if not finished:
                    # 如果没有进程结束，我们主动等待一段时间再检查
                    sleep(1)

            # 启动新进程
            p = Process(target=single_process, args=(batch, f'{base_path_idx}_{idx}'))
            p.start()
            process_list.append(p)
            print(p)

    # 等待所有进程完成
    for p in process_list:
        p.join()
