# FormulaReasoning: A Dataset for Formula-Based Numerical Reasoning

## FormulaReasoning
- train.json, 4608 questions
- id_test.json, 421 questions
- ood_test.json, 390 questions

## Requirements
- pytorch 2.0
- transformers
- zhipuai
- openai 0.28.0
- dashscope
  
Install numbat tool from [https://github.com/sharkdp/numbat].

## Baselines
#### LLMs
- `GLM4`: baselines/LLMs/GLM/ChatGLM4_api.py
- `GPT-3.5-turbo`: baselines/LLMs/GLM/ChatGPT_api.py
- `Qwen-max`: baselines/LLMs/GLM/Qwen_api.py
- `other LLMs`: download model files from huggingface and then `cd baselines/LLMs/ && python run.py --model_name_or_path /path/to/llm --data_file datas/id_test_zero_shot.json`. `data_file` could be one of `[id_test_zero_shot, ood_test_zero_shot, id_test_5_shot, ood_test_5_shot]`.
- eval: `cd baselines/LLMs/ && python eval_results.py --id_results {id_result_file} --ood_results {ood_result_file}`

#### Fine-tuned Small Models
- with calculator: `cd baselines/small_models && bash run_{model_type}.sh`
- without calculator: `cd baselines/small_models && bash run_{model_type}_wo_cal.sh`
  
model_type could be one of `[mt5_base, mt5_large, qwen]`


#### Formula Retriever
- train formula retriever: `cd baselines/RAG/ && bash run.sh`
- eval formula retriever: `cd baselines/RAG/ && python eval.py --model_path outputs_retriever`
