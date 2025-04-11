# modes.py
from utils import ds_error_info_process,ds_retrieve_sc_process

#------------------------------------------------------------self_edit
system_prompt_self_edit=\
"""You are a Solidity expert. Given an Solidity function with error information, your task is to rewrite the function to fix the error according to the original function. Put your code within code delimiters:
```solidity
# YOUR CODE HERE
```
For example:

# Given an Solidity function with error:
```solidity
// OpenSea OperatorFilterer
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    _setOperatorFilteringEnabled(_state);
}
```

# Error information:
Invalid solc compilation Error: Undeclared identifier. Did you mean "setOperatorFilteringEnabled"?
     |
2001 |         _setOperatorFilteringEnabled(_state);
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# According to the error, the statement `_setOperatorFilteringEnabled(_state);` has undeclared identifier. Therefore, your answer is:
```solidity 
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    operatorFilteringEnabled = _state;
}
```
""" 

def get_user_prompt_func_self_edit(d):
    error_infos=ds_error_info_process(d['error_infos'])
    
    user_prompt=\
f"""
# Given an Solidity function with error:
```solidity
{d['funitgen']}
```

# Error information:
{error_infos}

# According to the error, your answer is:
"""
    return user_prompt

#------------------------------------------------------------ra_self_edit
system_prompt_ra_self_edit=\
"""You are a Solidity expert. Given an Solidity function with error information, your task is to rewrite the function to fix the error according to the in-file code snippets of the function and the original function. Put your code within code delimiters:
```solidity
# YOUR CODE HERE
```
For example:

# Given an Solidity function with error:
```solidity
// OpenSea OperatorFilterer
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    _setOperatorFilteringEnabled(_state);
}
```

# Error information:
Invalid solc compilation Error: Undeclared identifier. Did you mean "setOperatorFilteringEnabled"?
     |
2001 |         _setOperatorFilteringEnabled(_state);
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# In-file code snippets of the function:
```solidity
(bool success, bytes memory returndata) = target.staticcall(data);

return functionStaticCall(target, data, "Address: low-level static call failed");

if (address(OPERATOR_FILTER_REGISTRY).code.length > 0 && operatorFilteringEnabled) {

bool public operatorFilteringEnabled = true;
```

# According to the error, the statement `_setOperatorFilteringEnabled(_state);` has undeclared identifier. According to the in-file code snippets, identifier `_setOperatorFilteringEnabled` should be replaced as `operatorFilteringEnabled`. Therefore, your answer is:
```solidity 
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    operatorFilteringEnabled = _state;
}
```
""" 

def get_user_prompt_func_ra_self_edit(d):
    error_infos=ds_error_info_process(d['error_infos'])
    retrieve_sc=ds_retrieve_sc_process(d['retrieve_sc'])
    
    user_prompt=\
f"""
# Given an Solidity function with error:
```solidity
{d['funitgen']}
```

# Error information:
{error_infos}

# In-file code snippets of the function:
```solidity
{retrieve_sc}
```

# According to the error and the in-file code snippets, your answer is:
"""
    return user_prompt

#------------------------------------------------------------self_repair
system_prompt_self_repair=\
"""You are a helpful programming assistant and an expert Solidity programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. Put your fixed program within code delimiters, for example: 
```solidity
# YOUR CODE HERE
```
For example:

# INCORRECT CODE
```solidity
// OpenSea OperatorFilterer
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    _setOperatorFilteringEnabled(_state);
}
```

# The code encountered the error:
Invalid solc compilation Error: Undeclared identifier. Did you mean "setOperatorFilteringEnabled"?
     |
2001 |         _setOperatorFilteringEnabled(_state);
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Explanation and Fixed Program
The error indicates that the function `_setOperatorFilteringEnabled` is not declared in the current contract or inherited contracts. You need to ensure that this function is either defined in your contract or inherited from a base contract.

Here’s a fixed version of the program, assuming the `_setOperatorFilteringEnabled` function is named `operatorFilteringEnabled` function within the contract:
```solidity 
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    operatorFilteringEnabled = _state;
}
```
""" 

def get_user_prompt_func_self_repair(d):
    error_infos=ds_error_info_process(d['error_infos'])
    user_prompt=\
f"""
# INCORRECT CODE
```solidity
{d['funitgen']}
```
# The code encountered the error:
{error_infos}

# Explanation and Fixed Program
"""
    return user_prompt

#------------------------------------------------------------ra_self_repair
system_prompt_ra_self_repair=\
"""You are a helpful programming assistant and an expert Solidity programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program according to the in-file code snippets of the function. Put your fixed program within code delimiters, for example: 
```solidity
# YOUR CODE HERE
```
For example:

# INCORRECT CODE
```solidity
// OpenSea OperatorFilterer
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    _setOperatorFilteringEnabled(_state);
}
```

# The code encountered the error:
Invalid solc compilation Error: Undeclared identifier. Did you mean "setOperatorFilteringEnabled"?
     |
2001 |         _setOperatorFilteringEnabled(_state);
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# In-file code snippets of the function:
```solidity
(bool success, bytes memory returndata) = target.staticcall(data);

return functionStaticCall(target, data, "Address: low-level static call failed");

if (address(OPERATOR_FILTER_REGISTRY).code.length > 0 && operatorFilteringEnabled) {

bool public operatorFilteringEnabled = true;
```

# Explanation and Fixed Program
The error indicates that the function `_setOperatorFilteringEnabled` is not declared in the current contract or inherited contracts. You need to ensure that this function is either defined in your contract or inherited from a base contract.

Here’s a fixed version of the program, according to the in-file code snippets, identifier `_setOperatorFilteringEnabled` should be replaced as `operatorFilteringEnabled`:
```solidity 
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    operatorFilteringEnabled = _state;
}
```
""" 

def get_user_prompt_func_ra_self_repair(d):
    error_infos=ds_error_info_process(d['error_infos'])
    retrieve_sc=ds_retrieve_sc_process(d['retrieve_sc'])
    
    user_prompt=\
f"""
# INCORRECT CODE
```solidity
{d['funitgen']}
```

# The code encountered the error:
{error_infos}

# In-file code snippets of the function:
```solidity
{retrieve_sc}
```

# Explanation and Fixed Program
"""
    return user_prompt

#------------------------------------------------------------self_refine
system_prompt_self_refine=\
"""You are an Solidity expert. Given an Solidity function with errors, your task is to analysis why the function is wrong, and fix the error. You can only fix the code within the function body. Please fix the Solidity code within code delimiters:
```solidity
# YOUR CODE HERE
```
For example:

# INCORRECT CODE
```solidity
// OpenSea OperatorFilterer
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    _setOperatorFilteringEnabled(_state);
}
```

Why is this code wrong?

# Explanation and Fixed Program
The provided code snippet appears to be a function from a Solidity smart contract that is intended to enable or disable operator filtering. However, without additional context, it's difficult to determine if the code is "wrong" in a functional sense. Here are a few potential issues or considerations that could be relevant:

1. **Missing Implementation of `_setOperatorFilteringEnabled`:** The function `_setOperatorFilteringEnabled(_state);` is called, but it is not defined in the provided snippet. If this function is not implemented elsewhere in the contract, it will lead to a compilation error.

Here’s a fixed version of the program, assuming the `_setOperatorFilteringEnabled` function is named `operatorFilteringEnabled` function within the contract:
```solidity 
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    operatorFilteringEnabled = _state;
}
```
""" 

def get_user_prompt_func_self_refine(d):
    user_prompt=\
f"""
# INCORRECT CODE
```solidity
{d['funitgen']}
```

Why is this code wrong?

# Explanation and Fixed Program
"""
    return user_prompt

#------------------------------------------------------------ra_self_refine
system_prompt_ra_self_refine=\
"""You are an Solidity expert. Given an Solidity function with errors, your task is to analysis why the function is wrong, and fix the error according to the in-file code snippets of the function. You can only fix the code within the function body. Please fix the Solidity code within code delimiters:
```solidity
# YOUR CODE HERE
```
For example:

# INCORRECT CODE
```solidity
// OpenSea OperatorFilterer
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    _setOperatorFilteringEnabled(_state);
}
```

# In-file code snippets of the function:
```solidity
(bool success, bytes memory returndata) = target.staticcall(data);

return functionStaticCall(target, data, "Address: low-level static call failed");

if (address(OPERATOR_FILTER_REGISTRY).code.length > 0 && operatorFilteringEnabled) {

bool public operatorFilteringEnabled = true;
```

Why is this code wrong?

# Explanation and Fixed Program
The provided code snippet appears to be a function from a Solidity smart contract that is intended to enable or disable operator filtering. However, without additional context, it's difficult to determine if the code is "wrong" in a functional sense. Here are a few potential issues or considerations that could be relevant:

1. **Missing Implementation of `_setOperatorFilteringEnabled`:** The function `_setOperatorFilteringEnabled(_state);` is called, but it is not defined in the provided snippet. If this function is not implemented elsewhere in the contract, it will lead to a compilation error.

Here’s a fixed version of the program, according to the in-file code snippets, identifier `_setOperatorFilteringEnabled` should be replaced as `operatorFilteringEnabled`:
```solidity 
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    operatorFilteringEnabled = _state;
}
```
""" 

def get_user_prompt_func_ra_self_refine(d):
    retrieve_sc=ds_retrieve_sc_process(d['retrieve_sc'])
    
    user_prompt=\
f"""
# INCORRECT CODE
```solidity
{d['funitgen']}
```

# In-file code snippets of the function:
```solidity
{retrieve_sc}
```

Why is this code wrong?

# Explanation and Fixed Program
"""
    return user_prompt

#------------------------------------------------------------self_debug
system_prompt_self_debug=\
"""You are an expert programming assistant.
""" 

def get_user_prompt_func_self_debug(d):
    error_infos=ds_error_info_process(d['error_infos'])

    one_shot=\
"""
The code above fails the error. You can only fix the code within the function body. Please fix the Solidity code within code delimiters:
```solidity
# YOUR CODE HERE
```

For example:

# Given an Solidity function with error:
// OpenSea OperatorFilterer
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    _setOperatorFilteringEnabled(_state);
}

# Error information:
Invalid solc compilation Error: Undeclared identifier. Did you mean "setOperatorFilteringEnabled"?
     |
2001 |         _setOperatorFilteringEnabled(_state);
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# According to the error, the statement `_setOperatorFilteringEnabled(_state);` has undeclared identifier. Therefore, your answer is:
```solidity 
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    operatorFilteringEnabled = _state;
}
```
"""

    user_prompt=\
f"""
# Error information:
{error_infos}

# According to the error, your answer is:
"""
    return one_shot+user_prompt

def get_user_prompt_func_self_debug_1(d):
    user_prompt=\
f"""
```solidity
{d['funitgen']}
```

Explain the Solidity code line by line.
"""

    return user_prompt

#------------------------------------------------------------ra_self_debug
system_prompt_ra_self_debug=\
"""You are an expert programming assistant.
"""

def get_user_prompt_func_ra_self_debug(d):
    error_infos=ds_error_info_process(d['error_infos'])
    retrieve_sc=ds_retrieve_sc_process(d['retrieve_sc'])
    
    one_shot=\
"""
The code above fails the error. You can only fix the code within the function body according to the in-file code snippets of the function. Please fix the Solidity code within code delimiters:
```solidity
# YOUR CODE HERE
```

For example:

# Given an Solidity function with error:
// OpenSea OperatorFilterer
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    _setOperatorFilteringEnabled(_state);
}

# Error information:
Invalid solc compilation Error: Undeclared identifier. Did you mean "setOperatorFilteringEnabled"?
     |
2001 |         _setOperatorFilteringEnabled(_state);
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# In-file code snippets of the function:
```solidity
(bool success, bytes memory returndata) = target.staticcall(data);

return functionStaticCall(target, data, "Address: low-level static call failed");

if (address(OPERATOR_FILTER_REGISTRY).code.length > 0 && operatorFilteringEnabled) {

bool public operatorFilteringEnabled = true;
```

# According to the error, the statement `_setOperatorFilteringEnabled(_state);` has undeclared identifier. According to the in-file code snippets, identifier `_setOperatorFilteringEnabled` should be replaced as `operatorFilteringEnabled`. Therefore, your answer is:
```solidity 
function setOperatorFilteringEnabled(bool _state) external onlyOwner {
    operatorFilteringEnabled = _state;
}
```
"""

    user_prompt=\
f"""
# Error information:
{error_infos}

# In-file code snippets of the function:
```solidity
{retrieve_sc}
```

# According to the error and the in-file code snippets, your answer is:
"""
    return one_shot+user_prompt

# -----------------------------------------------------------模式与对应的system prompt和get_user_prompt函数映射
modes = {
# -----------------------------------------------------------self_edit
    'self_edit': {
        'system_prompt': system_prompt_self_edit,
        'get_user_prompt_func': get_user_prompt_func_self_edit,
    },
    
    'ra_self_edit': {
        'system_prompt': system_prompt_ra_self_edit,
        'get_user_prompt_func': get_user_prompt_func_ra_self_edit,
    },
# -----------------------------------------------------------self_repair
    'self_repair': {
        'system_prompt': system_prompt_self_repair,
        'get_user_prompt_func': get_user_prompt_func_self_repair,
    },
    
    'ra_self_repair': {
        'system_prompt': system_prompt_ra_self_repair,
        'get_user_prompt_func': get_user_prompt_func_ra_self_repair,
    },
# -----------------------------------------------------------self_refine
    'self_refine': {
        'system_prompt': system_prompt_self_refine,
        'get_user_prompt_func': get_user_prompt_func_self_refine,
    },
    
    'ra_self_refine': {
        'system_prompt': system_prompt_ra_self_refine,
        'get_user_prompt_func': get_user_prompt_func_ra_self_refine,
    },
# -----------------------------------------------------------self_debug
    'self_debug': {
        'system_prompt': system_prompt_self_debug,
        'get_user_prompt_func_1': get_user_prompt_func_self_debug_1,
        'get_user_prompt_func_2': get_user_prompt_func_self_debug,
    },
    
    'ra_self_debug': {
        'system_prompt': system_prompt_ra_self_debug,
        'get_user_prompt_func_1': get_user_prompt_func_self_debug_1,
        'get_user_prompt_func_2': get_user_prompt_func_ra_self_debug,
    },
}