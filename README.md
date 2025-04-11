# SolBench: A Dataset and Benchmark for Evaluating Functional Correctness in Solidity Code Completion and Repair    


## Installation
```
git clone git@github.com:ZaoyuChen/SolBench.git
cd SolBench
pip install -r requirements.txt
```
### Installation of Diffusc and Echidna
TODO:
<!-- 采样文件必须在diffusc文件夹下 -->
## Usage
### Code Completion
Using the following command to run code completion task on SolBench with different models and context lengths:
```
python src/code_completion/run.py --model_path your_model_path --context_length 256 --completions_save_path sample_results/code_completion/context_length_256.jsonl
```
`context_length` could be `0,256,512,1k,2k,4k`.

Then post process the model sampling results for executor evaluation:
```
python src/utils/post_process.py --base_save_dir diffusc/sample_results/code_completion/context_length_256 --sol_prompt_sc_fname data/SolBench_ds.parquet --openai_outputfile sample_results/code_completion/context_length_256.jsonl --model_type qwen
```
### Executor for Functional Correctness
Run the multi-process version of Diffusc and Echidna to evaluate the functional correctness of completed functions:
```
python src/executor/echidna_deploy_mp.py --base_path diffusc/sample_results/code_completion/context_length_256
```
Using the following to compute the pass rate:
```
python src/executor/PassRateStat_echidna.py --base_path diffusc/sample_results/code_completion/context_length_256
```
### Code Repair
First, extract the problems identified as functionally incorrect in the initial code completion and the feedback for the error:
```
python src/code_repair/extract_fail_ds.py
```
Then, perform code repair methods using:
```
python src/code_repair/run.py
```
You can specify code repair methods (self_edit,self_repair,self_refine) and context length in the parameter region.

Retrieve the sample results from OpenAI api:
```
python src/utils/openai_retrieval.py --save_dirname code_repair
```
For self_debug code repair method, due to its two-stage inference, using:
```
python src/code_repair/self_debug_1.py

python src/utils/openai_retrieval.py --save_dirname self_debug_1

python src/code_repair/self_debug_2.py

python src/utils/openai_retrieval.py --save_dirname self_debug_2
```
Finally, using the following to get the pass rate:
```
python src/utils/post_process.py --base_save_dir diffusc/sample_results/code_repair/context_length_256 --sol_prompt_sc_fname data/SolBench_ds.parquet --openai_outputfile sample_results/code_repair/context_length_256.jsonl --model_type openai

python src/executor/echidna_deploy_mp.py --base_path diffusc/sample_results/code_repair/context_length_256

python src/executor/PassRateStat_echidna.py --base_path diffusc/sample_results/code_repair/context_length_256
```
### Retrieval-Augmented Code Repair
First, extract the problems identified as functionally incorrect in the initial code completion and the feedback for the error, and also retrieve the relevant code snippets:
```
python src/retrieval_augmented_code_repair/extract_fail_ds_ra.py
```
You can specify retrieval methods (bm25 tfidf jaccard_sim unixcoder codebert) and context length in the parameter region. Here, we implement these retrieval methods from [CrossCodeEval](https://github.com/amazon-science/cceval).

For Longest Common Substring retrieval method, using:
```
python src/retrieval_augmented_code_repair/extract_fail_ds_lcs.py
```
Then, perform retrieval-augmented code repair methods using:
```
python src/retrieval_augmented_code_repair/run.py
```
You can specify retrieval-augmented code repair methods (ra_self_edit,ra_self_repair,ra_self_refine) and context length in the parameter region.

Retrieve the sample results from OpenAI api:
```
python src/utils/openai_retrieval.py --save_dirname ra_code_repair
```
For ra_self_debug retrieval-augmented code repair method, due to its two-stage inference, using:
```
python src/retrieval_augmented_code_repair/self_debug_1.py

python src/utils/openai_retrieval.py --save_dirname ra_self_debug_1

python src/code_repair/self_debug_2.py
****
python src/utils/openai_retrieval.py --save_dirname self_debug_2
```
Finally, using the following to get the pass rate:
```
python src/utils/post_process.py --base_save_dir diffusc/sample_results/ra_code_repair/context_length_256 --sol_prompt_sc_fname data/SolBench_ds.parquet --openai_outputfile sample_results/ra_code_repair/context_length_256.jsonl --model_type openai

python src/executor/echidna_deploy_mp.py --base_path diffusc/sample_results/ra_code_repair/context_length_256

python src/executor/PassRateStat_echidna.py --base_path diffusc/sample_results/ra_code_repair/context_length_256
```