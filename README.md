# FormulaReasoning: A Dataset for Formula-Based Numerical Reasoning

## FormulaReasoning

### Released Dataset
- `data/dataset_zh/train.json`, 4608 questions
- `data/dataset_zh/HoF_test.json`, 395 questions
- `data/dataset_zh/HeF_test.json`, 398 questions

The corresponding English version is located in `data/dataset_en`.

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
- eval: `cd baselines/LLMs/ && python eval_results.py --hof_results {HoF_result_file} --hef_results {HeF_result_file}`

#### DPO
We use MCTS to construct preference data, with the code provided in `baselines/MCTS-PRM`. The code for DPO can be found in `baselines/DPO`.

#### Fine-tuned Small Models
- with calculator: `cd baselines/small_models && bash run_qwen.sh`
- without calculator: `cd baselines/small_models && bash run_qwen_wo_cal.sh`


#### Formula Retriever
- train formula retriever: `cd baselines/RAG/ && bash run.sh`
- eval formula retriever: `cd baselines/RAG/ && python eval.py --model_path outputs_retriever`
