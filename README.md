# FormulaReasoning: A Dataset for Formula-Based Numerical Reasoning

## FormulaReasoning

### Released Dataset
- `data/dataset_zh/train.json`, 4608 questions
- `data/dataset_zh/id_test.json`, 421 questions
- `data/dataset_zh/ood_test.json`, 391 questions

The corresponding English version is located in `/data/dataset_en`.

## Requirements
- pytorch 2.0
- transformers
- zhipuai
- openai 0.28.0
- dashscope
  
Install numbat tool from [https://github.com/sharkdp/numbat].

## Baselines
#### LLMs
- `GLM-4` series: baselines/LLMs/GLM/ChatGLM4_api.py
- `GPT` series: baselines/LLMs/GLM/ChatGPT_api.py
- `Qwen` series: baselines/LLMs/GLM/Qwen_api.py
- `other LLMs`: download model files from huggingface and then `cd baselines/LLMs/ && python run.py --model_name_or_path /path/to/llm --data_file datas/id_test_zero_shot.json`. `data_file` could be one of `[id_test_zero_shot, ood_test_zero_shot, id_test_5_shot, ood_test_5_shot]`.
- eval: `cd baselines/LLMs/ && python eval_results.py --id_results {id_result_file} --ood_results {ood_result_file}`

#### Fine-tuned Small Models
- with calculator: `cd baselines/small_models && bash run_qwen.sh`
- without calculator: `cd baselines/small_models && bash run_qwen_wo_cal.sh`


#### Formula Retriever
- train formula retriever: `cd baselines/RAG/ && bash run.sh`
- eval formula retriever: `cd baselines/RAG/ && python eval.py --model_path outputs_retriever`
