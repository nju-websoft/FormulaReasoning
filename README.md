# FormulaReasoning: A Dataset for Formula-Based Numerical Reasoning

## FormulaReasoning

### Released Dataset
- `data/train.json`, 3958 questions
- `data/HoF_test.json`, 410 questions
- `data/HeF_test.json`, 383 questions
- `formulas.json`, formula database

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
