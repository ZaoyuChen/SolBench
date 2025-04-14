import torch, os, random,shutil,re,json
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def initialize_model(model_path,device):
    config = AutoConfig.from_pretrained(model_path)
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config).to(device)
    model.eval()
    return model

def initialize_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'
    tokenizer.add_bos_token=False
    tokenizer.add_eos_token=False
    return tokenizer


def truncate_funit(funit):
    '''
    包含了function这一行末尾的所有字符，除了换行符
    会找到{的位置，并将{之后的内容抹去
    返回的half_funit是\r\n换行格式
    '''
    half_funit=None
    ps=funit.splitlines()
    function_flag=False
    for idx,p in enumerate(ps):
        # 第一步，找到function开头
        if p.strip().startswith('function'):
            function_flag=True
            
        # 第二步，找到{的位置
        if (function_flag==True) and '{' in p:
            half_funit='\r\n'.join(ps[:idx])
            # 将包含第一个{的行的末尾清除干净
            p_find=p[:p.find('{')+1]
            # 防止half_funit为空时，多了一个换行符
            if half_funit != '':
                half_funit=half_funit+'\r\n'+p_find
            else:
                half_funit=p_find
            break
        
    return half_funit

def load_dataset_from_file_or_folder(file_folder_path):
    # 给定的路径是目录，从文件夹中加载parquet文件
    ds=None
    if os.path.isdir(file_folder_path):
        parquet_files=[]
        for filepath,dirnames,filenames in os.walk(file_folder_path):
            for filename in filenames:
                if filename.endswith('parquet'):
                    fullname = os.path.join(filepath, filename)
                    parquet_files.append(fullname)
        ds = load_dataset("parquet", data_files=parquet_files)['train']

    # 给定的路径是文件名，从文件中加载parquet文件
    elif os.path.isfile(file_folder_path):
        ds = load_dataset("parquet", data_files=file_folder_path)['train']
        
    # 可能没有添加parquet，添加了再load
    elif not file_folder_path.endswith('parquet'):
        print('尝试添加parquet后缀')
        ds=load_dataset_from_file_or_folder(file_folder_path+'.parquet')
        
    if ds is None:
        raise Exception(f'check the dataset path: {file_folder_path}')
    else:
        return ds

def change_line_break(to_change_str):
    """
    将字符串的换行符统一为\r\n
    """
    to_change_strps=to_change_str.splitlines()
    changed_str='\r\n'.join(to_change_strps)
    return changed_str

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
       
def pre_save_solfile(sol_name,save_dir):
    """
    sol_name 是sol文件的绝对路径
    save_dir 是sol文件路径的数字层级，origin前的路径
    """
    origin_dir=os.path.join(save_dir,'origin')
    origin_sol=os.path.join(origin_dir,sol_name.split('/')[-1])
    if not os.path.exists(origin_sol):
        os.makedirs(origin_dir,exist_ok=True)
        shutil.copy(sol_name,origin_dir)

def pre_save_solfile_DISL(source_code,save_dir,sol_name):
    """
    source_code 是sol文件的内容
    save_dir 是sol文件路径的数字层级，origin前的路径
    sol_name 是sol文件的合约名
    """
    origin_dir=os.path.join(save_dir,'origin')
    origin_sol=os.path.join(origin_dir,f'{sol_name}.sol')
    if not os.path.exists(origin_sol):
        os.makedirs(origin_dir,exist_ok=True)
    with open(origin_sol,'w') as f:
        f.write(source_code)

def post_process_and_save(args,text_out,funit,sc_ba,prompt_i,sol_name,save_dir):
    """
    text_out shape[0] 是为multi-sample准备的，对于贪婪采样，目前是1 
    """
    post_process_text_out=[]

    for text in text_out:
        # 将补全的函数体尾巴切干净
        text_truncate = brace_truncate(text)
        post_process_text_out.append(text_truncate)

    #replace the origin function as the generated one in the contract.
    for text_id,text in enumerate(post_process_text_out):
        half_funit=truncate_funit(funit)
        funit_completion=f'{half_funit}\r\n{text}'
        sc_before=sc_ba[0]
        sc_after=sc_ba[1]
        replaces=f'{sc_before}\r\n{funit_completion}\r\n{sc_after}'

        #save generated smart contracts
        sc_save=f'{args.model_name}_epoch{args.ckpt_id}_prompt{prompt_i}_sample{text_id}'
        sc_save_dir=os.path.join(save_dir,'test',sc_save)
        os.makedirs(sc_save_dir,exist_ok=True)
        with open(os.path.join(sc_save_dir,sol_name.split('/')[-1]),'w') as f:
            f.write(replaces)
    
    #save generated functions
    fn_save_dir=os.path.join(save_dir,'sample_functions')
    os.makedirs(fn_save_dir,exist_ok=True)
    fn_sample_save=f'{fn_save_dir}/{args.model_name}_epoch{args.ckpt_id}_prompt{prompt_i}.txt'
    with open(fn_sample_save,'w') as f:
        f.write(f'funit:\n\n{funit}\n\n')
        for i,text in enumerate(post_process_text_out):
            f.write(f'out[{i+1}]:\n\n{half_funit}\r\n{text}\n\n')

def post_process_and_save_DISL(args,text_out,funit,sc_ba,prompt_i,sol_name,save_dir):
    """
    text_out shape[0] 是为multi-sample准备的，对于贪婪采样，目前是1 
    """
    post_process_text_out=[]

    for text in text_out:
        # 将补全的函数体尾巴切干净
        text_truncate = brace_truncate(text)
        post_process_text_out.append(text_truncate)

    #replace the origin function as the generated one in the contract.
    for text_id,text in enumerate(post_process_text_out):
        half_funit=truncate_funit(funit)
        funit_completion=f'{half_funit}\r\n{text}'
        sc_before=sc_ba[0]
        sc_after=sc_ba[1]
        replaces=f'{sc_before}\r\n{funit_completion}\r\n{sc_after}'

        #save generated smart contracts
        sc_save=f'{args.model_name}_epoch{args.ckpt_id}_prompt{prompt_i}_sample{text_id}'
        sc_save_dir=os.path.join(save_dir,'test',sc_save)
        os.makedirs(sc_save_dir,exist_ok=True)
        with open(os.path.join(sc_save_dir,f'{sol_name}.sol'),'w') as f:
            f.write(replaces)
    
    #save generated functions
    fn_save_dir=os.path.join(save_dir,'sample_functions')
    os.makedirs(fn_save_dir,exist_ok=True)
    fn_sample_save=f'{fn_save_dir}/{args.model_name}_epoch{args.ckpt_id}_prompt{prompt_i}.txt'
    with open(fn_sample_save,'w') as f:
        f.write(f'funit:\n\n{funit}\n\n')
        for i,text in enumerate(post_process_text_out):
            f.write(f'out[{i+1}]:\n\n{half_funit}\r\n{text}\n\n')


def select_30_in_order(lst):
    """
    list小于30返回list
    list大于30，随机选出list 30个返回
    """
    if len(lst) <= 30:
        return lst  # 如果列表长度小于等于30，则直接返回原列表
    else:
        # 随机选择30个不同的索引
        indices = sorted(random.sample(range(len(lst)), 30))
        # 根据选中的索引获取元素
        selected_elements = [lst[i] for i in indices]
        return selected_elements

def return_func_body_from_funit(funit):
    """
    从funit切割掉half_funit，返回func_body
    """
    # 确认half_funit是function开头
    half_funit=truncate_funit(funit)
    if half_funit is None:
        # print("-" * 40)
        # print(f'funit:\n{funit}')
        raise Exception('half_funit is None')
    
    # 确认返回正确的func_body
    funit=change_line_break(funit)
    if half_funit in funit:
        func_body=funit.replace(half_funit,'')
        return func_body
    else:
        raise Exception('half_funit not in funit')

def return_func_sig_from_funit(funit):
    """
    从funit取出 half_funit ，从 half_funit 返回 函数头
    """
    half_funit=truncate_funit(funit)
    return return_func_sig_from_half_funit(half_funit)

def return_func_sig_from_half_funit(half_funit):
    """
    从 half_funit 返回 函数头
    """
    
    func_half_funit=None
    ps=half_funit.splitlines()
    for idx,p in enumerate(ps):
        # 第一步，找到function开头
        if p.strip().startswith('function'):
            func_half_funit='\r\n'.join(ps[idx:])
    if func_half_funit is None:
        raise Exception('函数头为空')
    else:
        return func_half_funit


def return_returns_type_from_funit(funit):
    """
    从funit取出 函数头 ，从 函数头 返回 returns 的类型
    """
    func_half_funit=return_func_sig_from_funit(funit)
    if 'returns' not in func_half_funit:
        return None
    else:
        func_return=func_half_funit[func_half_funit.find('returns'):]
        return func_return[func_return.find('(')+1:func_return.rfind(')')]

def save_list_of_dir_as_parquet(list_of_dir,parquet_file_path):
    """
    # 把list of dir文件存为parquet，通过pandas
    """
    directory=os.path.dirname(parquet_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")
    df = pd.DataFrame(list_of_dir)
    df.to_parquet(parquet_file_path, engine='pyarrow')
    print(f'Save to {parquet_file_path}')
    
def return_prompt_i_from_error_info(error_info,search_pattern):
    """
    根据search_pattern从error_info
    匹配prompt_i时，search_pattern，
    匹配line_no时，search_pattern，r'^(\d+)'
    """
    match = re.search(search_pattern, error_info)
    if match:
        prompt = match.group(1)  # 这将返回 'prompt4155'
        return prompt
    else:
        raise Exception('没有匹配到prompt_i')     


def return_top_p_list(sorted_list,top_p):
    """
    sorted_list是list of 
    [code_snippet,softmax_score]
    """
    cumulative_prob = 0.0
    filtered_by_top_p = []
    for item in sorted_list:
        cumulative_prob += item[1]
        filtered_by_top_p.append(item)
        if cumulative_prob >= top_p:
            break
    return filtered_by_top_p

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
    return completion  # 如果没有找到这样的位置，返回 completion本身

def split_list_into_n(input_list, n):    
    """
    将输入的list均匀的分割成n份，返回切割好的list of list
    """
    k, m = divmod(len(input_list), n)
    return [input_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def read_jsonl(file_path):
    """
    读取JSON Lines文件
    返回list of json loads
    """
    ds=[]
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f:
            ds.append(json.loads(line))
    return ds

def write_jsonl(list_of_dict,file_path):
    """
    写入JSON Lines文件
    会根据 file_path 创建目录
    """
    directory=os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(file_path, 'w') as file:
        for item in list_of_dict:
            json.dump(item, file)
            file.write('\n')

def read_funit_gen_from_sample_txt(file_path):
    """
    从sample txt比如diffusc/sample/clean_2k_4omini_new_t0/00037/sample_functions/gpt-4o-mini-2024-07-18_epoch0_prompt1142.txt，读取funit gen
    """
    extracted_texts=[]
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    funit_marker = 'funit:\n\n'
    out_marker = '\n\nout[1]:\n\n'
    start_index = len(funit_marker)
    end_index = content.find(out_marker)
    funit = content[start_index:end_index]

    end_marker = '\n\n'
    start_index = end_index + len(out_marker)
    end_index = -len(end_marker)
    funit_gen = content[start_index:end_index]
    
    return funit,funit_gen
                
def read_funit_gen_from_sample_dir(base_path,prompt_i_search_pattern):
    """
    从sample dir比如/home/data3/zaoyu/diffusc/sample/clean_2k_4omini_new_t0，也就是sol num的父目录处，读取prompt i和对应的funit gen
    prompt_i_search_pattern用于从sample txt文件名提取prompt_i
    返回的prompt_i_funit是以prompt_i为键，funit_gen为值的字典
    """
    prompt_i_funit = {}

    for sol_num in os.listdir(base_path):
        for sample_txt in os.listdir(os.path.join(base_path, sol_num, 'sample_functions')):
            prompt_i=return_prompt_i_from_error_info(sample_txt,prompt_i_search_pattern)
            _, funit_gen = read_funit_gen_from_sample_txt(os.path.join(base_path, sol_num, 'sample_functions', sample_txt))
            prompt_i_funit[prompt_i]=funit_gen
    return prompt_i_funit

def read_diffusc_echidna_error_info_paths(base_path):
    """
    将base_path路径下的所有文件diffusc_echidna_error_info.json的路径返回
    """
    result_fs=[]
    for filepath in fast_walk(base_path):            
        if filepath.endswith('diffusc_echidna_result.json') and 'ipynb_checkpoints' not in filepath:
            result_fs.append(filepath)
    return result_fs

def read_error_infos_from_sample_dir(base_path,prompt_i_search_pattern):
    """
    从sample dir比如/home/data3/zaoyu/diffusc/sample/clean_2k_4omini_new_t0，也就是sol num的父目录处，读取prompt i和对应的error_infos
    prompt_i_search_pattern用于从错误信息提取prompt_i
    返回的prompt_i_errors是以prompt_i为键，append_errors为值的字典
    """
    prompt_i_errors={}
    sol_fs=read_diffusc_echidna_error_info_paths(base_path)

    # 错误文件循环
    for sol_f in tqdm(sol_fs):
        with open(sol_f,'r') as f:
            error_infos=json.load(f)
        
        # 跳过没有错误的
        if len(error_infos)==0:
            continue

        # 对单个错误文件里的每个prompt错误循环
        for value in error_infos.values():
            prompt_i=return_prompt_i_from_error_info(value, prompt_i_search_pattern)
            append_errors=[]
            if 'Invalid compilation: \n' in value:
                errors_info=value[
                    (value.find('Invalid compilation: \n')+len('Invalid compilation: \n')):value.rfind('\n\n\n\nTraceback (most recent call last):')
                ]
                errors=errors_info.split('\n\n')
                for error in errors:
                    if ('error' in error) or ('Error' in error):
                        append_errors.append(error)
            else:
                append_errors.append(value)
            prompt_i_errors[prompt_i]=append_errors
    return prompt_i_errors

def extract_prompt_i_from_error_infos_in_sample_dir(base_path,prompt_i_search_pattern):
    """
    在base_path sample_dir的所有错误信息里提取prompt_i，返回list of prompt_i
    """
    extract_idxs=[]
    sol_fs=read_diffusc_echidna_error_info_paths(base_path)
    # 错误文件循环
    for sol_f in sol_fs:
        with open(sol_f,'r') as f:
            error_infos=json.load(f)
        
        # 跳过没有错误的
        if len(error_infos)==0:
            continue

        # 对单个错误文件里的每个prompt错误循环
        for value in error_infos.values():
            # 提取prompt_i
            prompt_i=return_prompt_i_from_error_info(value, prompt_i_search_pattern)
            extract_idxs.append(prompt_i)
            if prompt_i is None:
                print(print_value)
                raise
    return extract_idxs

def char_ngrams(text, n):
    """
    返回字符串的字符级别n-gram
    # 示例
    text = "hello"
    n = 2
    print(char_ngrams(text, n))
    # 输出应为 ['he', 'el', 'll', 'lo']
    """
    return [text[i:i+n] for i in range(len(text)-n+1)]

def ds_error_info_process(list_of_error_infos):
    """
    将数据集的json化的list格式的error_infos去除文件路径，返回换行符连接的字符串形式的error_infos
    """
    error_infos=json.loads(list_of_error_infos)
    
    # 将error info中的文件路径删掉
    error_infos_prc=[]
    for error_info in error_infos:
        error_infops=error_info.splitlines()
        # 如果包含文件路径
        if error_infops[2].strip()=='|':
            error_infops.pop(1)
            error_info='\n'.join(error_infops)
            error_infos_prc.append(error_info)
            error_infos='\n\n'.join(error_infos_prc)
    return error_infos

def ds_retrieve_sc_process(list_of_retrieve_scs):
    """
    将数据集的json化的list格式的retrieve_scs去除前后空格，返回换行符连接的字符串形式的 retrieve_scs
    """
    retrieve_scs=json.loads(list_of_retrieve_scs)
    retrieve_scs=[retrieve_sc.strip() for retrieve_sc in retrieve_scs]
    retrieve_sc='\r\n\r\n'.join(retrieve_scs)
    return retrieve_sc

def model_call(messages):
    """
    openai chatgpt使用，默认模型是gpt-4o-mini，换模型需要在utils文件更换，输入是整个messages
    """
    model=["gpt-4o-mini","gpt-3.5-turbo",'gpt-4o'][0]
    client = OpenAI(
        api_key='your api key'
    )
    
    completions = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )            
    print(completions.choices[0].message.content)        
    return completions

def return_doc_sc(class_code,func_code):
    """
    根据完整的合约class_code，和只有signature和body的func_code，提取出comment和sc_ba (list)
    """
    try:
        # 提取func_documentation，返回func_documentation，返回删掉func_documentation和func_code的sc
        class_code_splits=class_code.split(func_code)
        assert len(class_code_splits) == 2, "长度不正确，应该是2"
        code_before=class_code_splits[0]
        code_after=class_code_splits[1]
        code_before_lines=code_before.splitlines()
        comment_in_flag=False
        # 提取func_documentation，得到删掉func_documentation的code_before
        for idx in range(len(code_before_lines) - 1, -1, -1):
            # 确认处于多行注释中
            if code_before_lines[idx].strip().endswith('*/'):
                comment_in_flag=True
                
            # 找到注释开头的位置 /*，结束查找
            if code_before_lines[idx].strip().startswith('/*'):            
                sc='\r\n'.join(code_before_lines[:idx])
                func_documentation_lines=code_before_lines[idx:]
                break
            # 找到注释//的上一行不是//的位置，结束查找
            if code_before_lines[idx].strip().startswith('//'):
                if code_before_lines[idx - 1].strip().startswith('//'):
                    continue
                else:
                    sc='\r\n'.join(code_before_lines[:idx])
                    func_documentation_lines=code_before_lines[idx:]
                    break

            # 既不在多行注释也没有注释符号，则没有注释
            if (comment_in_flag is False) and (code_before_lines[idx].strip() != ''):
                break
    
        # 拼接func_documentation_lines,包含了 func_code 前面的空格，与func_code直接拼接
        func_documentation='\r\n'.join(func_documentation_lines)
        
        # for idx in range(len(func_documentation_lines) - 1, -1, -1):
        #     if func_documentation_lines[idx].strip() != '':
        #         func_documentation='\r\n'.join(func_documentation_lines[:idx+1])
        #         break
            
    # 函数没有注释，文件给的注释是上一个语句的文尾注释. func_documentation_lines UnboundLocalError
    except UnboundLocalError:
        # print('UnboundLocalError')
        # print('class_code******************************')
        # print(class_code)
        # print('func_code******************************')
        # print(func_code)
        return None,None
    
    except AssertionError:
        # print('AssertionError')
        # print('class_code******************************')
        # print(class_code)
        # print('func_code******************************')
        # print(func_code)
        return None,None
    
    return func_documentation, [sc,code_after]

def line_break_unify(original_string):
    # 将字符串的换行符统一为\r\n
    modified_string = '\r\n'.join(original_string.splitlines())
    return modified_string

def fast_walk(base_path):
    """
    逐个返回路径下的每一个文件名，遍历
    """
    for entry in os.scandir(base_path):
        if entry.is_dir():
            yield from fast_walk(entry.path)
        else:
            yield entry.path
            
def cal_passrate(base_path):
    """
    统计并且打印采样正确率
    """
    # 统计采样个数
    txt_fs=[]
    # 有验证结果的sol num个数
    result_fs=[]
    
    pbar = tqdm(total=2609128)
    for filepath in fast_walk(base_path):
        
        if filepath.endswith('.txt') and 'ipynb_checkpoints' not in filepath:
            fullname = os.path.join(filepath)
            txt_fs.append(fullname)
            pbar.update(1)
            
        if filepath.endswith('diffusc_echidna_result.json') and 'ipynb_checkpoints' not in filepath:
            result_fs.append(filepath)
    pbar.close()
    
    num_prompt=len(txt_fs)/1
    print(base_path)
    print("-" * 40)
    
    print(f'采样个数len(txt_fs)={len(txt_fs)},num_prompt={num_prompt}')
    print("-" * 40)
    
    print(f'有验证结果的sol num个数len(result_fs)={len(result_fs)}, assert {len(result_fs)}={len(os.listdir(base_path))}')
    print("-" * 40)
        
    for pass_compile_flag in ['pass','compile']:
    # for pass_compile_flag in ['pass']:
        pass_stats=[]
        for result_f in tqdm(result_fs):
            with open(result_f,'r') as f:
                a=json.load(f)
                
            if pass_compile_flag=='pass':
                if 'pass' in a:
                    b=a['pass']
                else:
                    continue
            elif pass_compile_flag=='compile':
                if 'compile' in a:
                    b=a['compile']
                else:
                    continue
            
            addres=[]
            for c in b:
                addre={}
                c=c.split('_')
                for ci in c:
                    if 'epoch' in ci:
                        addre['epoch']=ci.lstrip('epoch')
                    elif 'prompt' in ci:
                        addre['prompt']=ci.lstrip('prompt')
                    elif 'sample' in ci:
                        addre['sample']=ci.lstrip('sample')
                    else:
                        addre['model']=addre['model']+f'_{ci}' if addre.get('model') else ci
                addres.append(addre)
                
            from collections import defaultdict
            pass_stat={}
            for addre in addres:
                pass_stat[addre['model']]={}
            for addre in addres:
                pass_stat[addre['model']]['epoch_{}'.format(addre['epoch'])]=defaultdict(int)
            for addre in addres:
                model=addre['model']
                epoch=addre['epoch']
                prompt=addre['prompt']
                pass_stat[model][f'epoch_{epoch}'][f'prompt_{prompt}']+=1
            pass_stats.append(pass_stat)
            
        # sum of all files to calculate the pass rate.
        pass_rate={}
        for pass_stat in pass_stats:
            for model,model_dict in pass_stat.items():
                pass_rate[model]=defaultdict(int)
        for pass_stat in pass_stats:
            for model,model_dict in pass_stat.items():
                for epoch,epoch_dict in model_dict.items():
                    # 只计数了存在prompt i的条目，没有计算通过几次，也就是pass@n
                    pass_rate[model][epoch]+=len(epoch_dict)
        for model,model_dict in pass_rate.items():
            for epoch,epoch_dict in model_dict.items():
                pass_rate[model][epoch]=format((pass_rate[model][epoch]/num_prompt*100),'.2f')
                
        
        print(pass_compile_flag)
        pass_rate_json=json.dumps(pass_rate, indent=4,sort_keys=True)
        print(pass_rate_json)
        print("-" * 40)
    return {'pass_compile_flag':pass_compile_flag,'pass_rate':pass_rate}

def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f'{key}:\n')
        print(f'{value}\n')
        print('#'*50)
