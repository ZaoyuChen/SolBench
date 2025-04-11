"""
调用各种检索方法，输入funitgen,sc,error_infos，确定query，返回检索的代码片段
"""

from crosscodeeval_methods import lexical_ranking,semantic_ranking
import re,logging

def return_query_code_snippet_from_error_info(error_infos,funitgen):
    """
    将作为query的变量名，错误代码行，错误函数提取出来，并去重
    """
    # 确定需要检索的代码片段
    all_query_code_snippets=[]    
    
    # 多个错误信息循环
    for error_info in error_infos:
        error_infops=error_info.splitlines()
        
        # # 确定是不是Undeclared identifier error
        # if 'Undeclared identifier' in error_info:
        #     start_idx=error_infops[-1].find('^')
        #     end_idx=error_infops[-1].rfind('^')
        #     identifier=error_infops[-2][start_idx:end_idx+1]
        #     all_query_code_snippets.append(identifier)
            
        # 检查错误是不是给出错误代码行
        if error_infops[2].strip()=='|':
            # 使用正则表达式寻找数字后跟'|'的所有内容
            pattern = r'\d+\s*\|\s*(.*)'
            match = re.search(pattern, error_infops[3])
            if match:
                # 提取并打印匹配到的内容
                error_code = match.group(1)
                
                all_query_code_snippets.append(error_code)
            else:
                print(error_info)
                raise Exception('错误信息给出代码行，但提取失败')
        
        # 错误没有给出错误代码行
        if (error_infops[2].strip()!='|') or all_query_code_snippets==['']:
            logging.info('错误信息没有给出错误代码行')
            all_query_code_snippets.append(funitgen)
    
    # 去重并且去掉空字符串
    all_query_code_snippets = [item for item in all_query_code_snippets if item != ""]
    all_query_code_snippets=set(all_query_code_snippets)
    return all_query_code_snippets

# 根据需要检索的变量名检索代码片段
# 优先使每个变量被检索，再循环检索每个变量
def retrieve_code_snippet(all_query_code_snippets,sc,retrieve_num,retrieve_len,crosscodeeval_method,crosscodeeval_method_name):
    sc_lines=sc.splitlines()
    # 滑动步数
    sliding_step = 1
    # 使用滑动窗口按行数切分
    sc_lines_splits = [sc_lines[i:i+retrieve_len] for i in range(0, len(sc_lines) - retrieve_len + 1, sliding_step)]

    # 一次只会返回一个查找位置，或者没找到
    def retrieve_for_single_query(sc_lines_splits,query,crosscodeeval_method,crosscodeeval_method_name):
        documents=['\n'.join(sc_lines_split) for sc_lines_split in sc_lines_splits]
        # 去掉空白documents
        documents=[document for document in documents if document.strip() != '']
        document_ids = range(len(documents))
        
        docs, doc_ids, scores=crosscodeeval_method(query,documents,crosscodeeval_method_name,doc_ids=document_ids,score_threshold=0.0001)
        
        # 跳过代码是注释
        filter_docs=[]
        filter_doc_ids=[]
        for doc_idx,doc in enumerate(docs):
            if (doc.strip().startswith('/*')) or (doc.strip().startswith('*')) or (doc.strip().startswith('//')):
                continue
            else:
                filter_docs.append(doc)
                filter_doc_ids.append(doc_ids[doc_idx])

        # 返回 retrieve_num 个 doc
        return filter_docs[:retrieve_num],filter_doc_ids[:retrieve_num]

    # 对需要检索的变量循环
    all_docs=[]
    all_doc_ids=[]
    for query in all_query_code_snippets:
        docs, doc_ids=retrieve_for_single_query(sc_lines_splits,query,crosscodeeval_method,crosscodeeval_method_name)
        all_docs.extend(docs)
        all_doc_ids.extend(doc_ids)
        
    return all_docs, all_doc_ids

# 根据代码片段idx 和检索长度返回检索代码片段,返回检索代码片段的list
def from_idx_to_code_snippets(all_doc_ids,sc,retrieve_len):
    sc_lines=sc.splitlines()
    # 滑动步数
    sliding_step = 1
    # 使用滑动窗口按行数切分
    sc_lines_splits = [sc_lines[i:i+retrieve_len] for i in range(0, len(sc_lines) - retrieve_len + 1, sliding_step)]
    
    documents=['\n'.join(sc_lines_split) for sc_lines_split in sc_lines_splits]
    # 去掉空白documents
    documents=[document for document in documents if document.strip() != '']
    retrieve_sc=[]

    # 去重并且升序排列
    all_doc_ids=sorted(set(all_doc_ids))

    retrieve_sc_list = [documents[all_doc_id] for all_doc_id in all_doc_ids]
    return retrieve_sc_list

def main(funitgen,sc,error_infos,retrieve_num,retrieve_len,crosscodeeval_method_name):
    # 确定需要检索的代码片段
    all_query_code_snippets=return_query_code_snippet_from_error_info(error_infos,funitgen)
    # print(all_query_code_snippets)
    # 根据需要检索的代码片段检索,得到代码片段idx
    if crosscodeeval_method_name in ['bm25','tfidf','jaccard_sim',]:
        crosscodeeval_method=lexical_ranking
    if crosscodeeval_method_name in ['unixcoder','codebert',]:
        crosscodeeval_method=semantic_ranking
    
    all_docs, all_doc_ids=retrieve_code_snippet(all_query_code_snippets,sc,retrieve_num,retrieve_len,crosscodeeval_method,crosscodeeval_method_name)

    # 根据代码片段idx 和检索长度返回检索代码片段
    retrieve_sc=from_idx_to_code_snippets(all_doc_ids,sc,retrieve_len)

    return retrieve_sc