"""
根据给定的 funit 解析出需要检索的变量
根据变量在上下文中检索出对应的代码片段
"""
from solidity_parser import parser
import re,sys,io
import logging
from utils import char_ngrams
# 内置变量，不用检索
variable_built_in=[
    # address
    'address', 'balance', 'transfer', 'send', 'call', 'callcode', 'delegatecall',
    # block：
    'block', 'coinbase', 'difficulty', 'gaslimit', 'number', 'timestamp','basefee','blobbasefee', 'chainid','prevrandao',   
    # msg：
    'msg', 'data', 'gas', 'sender', 'sig', 'value',
    # tx：
    'tx', 'gasprice', 'origin',
    # 数学
    'addmod', 'mulmod', 'keccak256', 'sha256', 'sha3', 'ripemd160', 'ecrecover',
    # abi
    'abi', 'encode', 'encodePacked', 'encodeWithSelector', 'encodeWithSignature', 'decode','encodeCall',
    # 数据类型
    # 'bool'
    'bool', 'true', 'false', 
    # 'uint',
    'uint', 'uint8', 'uint16', 'uint24', 'uint32', 'uint40', 'uint48', 'uint56', 'uint64', 'uint72', 'uint80', 'uint88', 'uint96', 'uint104', 'uint112', 'uint120', 'uint128', 'uint136', 'uint144', 'uint152', 'uint160', 'uint168', 'uint176', 'uint184', 'uint192', 'uint200', 'uint208', 'uint216', 'uint224', 'uint232', 'uint240', 'uint248', 'uint256',
    # 'int', 
    'int', 'int8', 'int16', 'int24', 'int32', 'int40', 'int48', 'int56', 'int64', 'int72', 'int80', 'int88', 'int96', 'int104', 'int112', 'int120', 'int128', 'int136', 'int144', 'int152', 'int160', 'int168', 'int176', 'int184', 'int192', 'int200', 'int208', 'int216', 'int224', 'int232', 'int240', 'int248', 'int256',
    # 'bytes', 
    'bytes', 'bytes1', 'bytes2', 'bytes3', 'bytes4', 'bytes5', 'bytes6', 'bytes7', 'bytes8', 'bytes9', 'bytes10', 'bytes11', 'bytes12', 'bytes13', 'bytes14', 'bytes15', 'bytes16', 'bytes17', 'bytes18', 'bytes19', 'bytes20', 'bytes21', 'bytes22', 'bytes23', 'bytes24', 'bytes25', 'bytes26', 'bytes27', 'bytes28', 'bytes29', 'bytes30', 'bytes31', 'bytes32',
    # 其他:
    'blockhash', 'gasleft', 'now','this', 'selfdestruct', 'suicide','blobhash',
    # 错误处理
    'assert', 'require', 'revert', 
]

# 给定sourceUnit，提取出变量名
def extract_names(node):
    names = []
    
    if isinstance(node, dict):
        for key, value in node.items():
            # memberName
            if isinstance(value, (dict, list)):
                names.extend(extract_names(value))
            elif (key == 'name') or (key == 'memberName'):
                names.append(value)
                
    elif isinstance(node, list):
        for item in node:
            names.extend(extract_names(item))
    
    return names

def extract_names_for_code_line(node):
    names = []
    
    if isinstance(node, dict):
        for key, value in node.items():
            # memberName
            if isinstance(value, (dict, list)):
                names.extend(extract_names_for_code_line(value))
            elif (key == 'namePath') or (key == 'memberName'):
                names.append(value)
                
    elif isinstance(node, list):
        for item in node:
            names.extend(extract_names_for_code_line(item))
    
    return names

def parser_retrieve_variable_for_line(code_line,funit):
    retrieve_variables=[]
    
    # 代码行的变量
    sourceUnit = parser.parse(code_line, loc=False,strict=True)
    variable_in_code_line=extract_names_for_code_line(sourceUnit)
    
    # 函数头的变量
    try:
        sourceUnit = parser.parse(funit, loc=False,strict=True)
        variable_in_signature=extract_names(sourceUnit.children[0].parameters)
        variable_in_signature.extend(extract_names(sourceUnit.children[0].returnParameters))
        variable_in_signature=set(variable_in_signature)
    except:
        variable_in_signature=[]
    # 筛选出需要检索的变量名
    for variable in variable_in_code_line:
        # 存在检索到 None 的变量名情况
        if (variable not in variable_in_signature) and (variable not in variable_built_in) and (variable is not None):
            retrieve_variables.append(variable)
            
    return retrieve_variables

# 确定需要检索的变量名
def parser_retrieve_variable(funit):
    retrieve_variables=[]
    sourceUnit = parser.parse(funit, loc=False,strict=True)
    
    # 函数体变量名
    variable_in_body=extract_names(sourceUnit.children[0].body)
    variable_in_body=set(variable_in_body)
    # 函数头变量名
    variable_in_signature=extract_names(sourceUnit.children[0].parameters)
    variable_in_signature.extend(extract_names(sourceUnit.children[0].returnParameters))
    variable_in_signature=set(variable_in_signature)

    # 筛选出需要检索的变量名
    for variable in variable_in_body:
        # 存在检索到 None 的变量名情况
        if (variable not in variable_in_signature) and (variable not in variable_built_in) and (variable is not None):
            retrieve_variables.append(variable)
            
    return retrieve_variables

# 根据需要检索的变量名检索代码片段
# 优先使每个变量被检索，再循环检索每个变量
def retrieve_code_snippet(retrieve_variables,sc,retrieve_num):
    
    retrieve_sc_idxs=[]
    sc_lines=sc.splitlines()
    
    # 一次只会返回一个查找位置，或者没找到
    def retrieve_for_single_variable(sc_lines,retrieve_variable,retrieve_sc_idxs):
        # 将变量分解为n-grams，查找单个n-gram
        for len_gram in range(len(retrieve_variable),1,-1):
            retrieve_variable_grams=char_ngrams(retrieve_variable,len_gram)
            for retrieve_variable_gram in retrieve_variable_grams:
                
                # 对上下文循环，从下往上找，定位变量位置
                # for idx in range(len(sc_lines) - 1, -1, -1):
                # 从上往下
                for idx in range(len(sc_lines)):
                    sc_line=sc_lines[idx]
                    
                    if retrieve_variable_gram in sc_line:
                        # 如果当前代码是注释， 就跳过
                        if (sc_line.strip().startswith('/*')) or (sc_line.strip().startswith('*')) or (sc_line.strip().startswith('//')):
                            continue
                        # 检查idx是否已经在retrieve_sc_idxs里
                        if idx not in retrieve_sc_idxs:
                            retrieve_sc_idxs.append(idx)
                            return retrieve_sc_idxs
                        else: 
                            continue
        return retrieve_sc_idxs
    
    # 对每个变量的检索次数循环
    for _ in range(retrieve_num):   
        # 对需要检索的变量循环
        for retrieve_variable in retrieve_variables:
            retrieve_sc_idxs=retrieve_for_single_variable(sc_lines,retrieve_variable,retrieve_sc_idxs)
            
    return retrieve_sc_idxs

# 根据代码片段idx 和检索长度返回检索代码片段
def from_idx_to_code_snippets(retrieve_sc_idxs,sc,retrieve_len):
    retrieve_sc=[]
    sc_lines=sc.splitlines()
    
    def get_int_list_around_center(retrieve_sc_idx, retrieve_len):
        # 输出以retrieve_sc_idx为中心，长度为retrieve_len的整数列表
        half_len = retrieve_len // 2
        start_idx = retrieve_sc_idx - half_len
        end_idx = retrieve_sc_idx + half_len
        
        # 如果retrieve_len是奇数，我们希望中心索引也在结果列表中
        if retrieve_len % 2 != 0:
            end_idx += 1  # 因为range函数是左闭右开区间，所以这里end_idx加1来包含中心索引

        return list(range(start_idx, end_idx))

    def get_str_list_from_indices(str_list, indices):
        # 确保只选择那些在str_list有效范围内的索引，并且返回对应的字符串列表
        # 过滤掉超出str_list范围的索引
        valid_indices = [idx for idx in indices if 0 <= idx < len(str_list)]
        return [str_list[idx] for idx in valid_indices]

    # 去重
    retrieve_sc_idxs=set(retrieve_sc_idxs)
    for retrieve_sc_idx in retrieve_sc_idxs:
        # 获取整数列表
        int_list = get_int_list_around_center(retrieve_sc_idx, retrieve_len)

        # 根据整数列表获取对应的字符串列表
        result_str_list = get_str_list_from_indices(sc_lines, int_list)
        
        # 每个retrieve_sc_idx拼接一个对应的检索的code snippet
        retrieve_sc.append(
            '\r\n'.join(result_str_list)
        )

    return retrieve_sc

def return_variables_from_error_info(error_infos,funit,funitgen):
    # 确定需要检索的变量名
    all_retrieve_variables=[]    
    
    # 多个错误信息循环
    for error_info in error_infos:
        error_infops=error_info.splitlines()
        
        # 确定是不是Undeclared identifier error
        if 'Undeclared identifier' in error_info:
            start_idx=error_infops[-1].find('^')
            end_idx=error_infops[-1].rfind('^')
            identifier=error_infops[-2][start_idx:end_idx+1]
            all_retrieve_variables.append(identifier)
            
        # 检查错误是不是给出错误代码行
        elif error_infops[2].strip()=='|':
            # 使用正则表达式寻找数字后跟'|'的所有内容
            pattern = r'\d+\s*\|\s*(.*)'
            match = re.search(pattern, error_infops[3])
            if match:
                # 提取并打印匹配到的内容
                error_code = match.group(1)
                
                retrieve_variables=parser_retrieve_variable_for_line(error_code,funit)
                all_retrieve_variables.extend(retrieve_variables)
            else:
                print(error_info)
                raise Exception('错误信息给出代码行，但提取失败')
        
        # 错误没有给出错误代码行
        else:
            logging.info('错误信息没有给出错误代码行')
            all_retrieve_variables=parser_retrieve_variable(funitgen)
    
    # 去重
    all_retrieve_variables=set(all_retrieve_variables)
    return all_retrieve_variables

def main(funit,funitgen,sc,error_infos,retrieve_num,retrieve_len):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    all_retrieve_variables=return_variables_from_error_info(error_infos,funit,funitgen)
    
    # 根据需要检索的变量名检索,得到代码片段idx
    retrieve_sc_idxs=retrieve_code_snippet(all_retrieve_variables,sc,retrieve_num)
    
    # 根据代码片段idx 和检索长度返回检索代码片段
    retrieve_sc=from_idx_to_code_snippets(retrieve_sc_idxs,sc,retrieve_len)
    # print(f'len(retrieve_sc)={len(retrieve_sc)},len(retrieve_variables)={len(retrieve_variables)}')
    return retrieve_sc