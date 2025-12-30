# FormulaReasoning: A Dataset for Formula-Based Numerical Reasoning

## FormulaReasoning

### Released Dataset

#### FormulaReasoning
- `data/FormulaReasoning/train.json`, 4524 questions
- `data/FormulaReasoning/HoF_test.json`, 413 questions
- `data/FormulaReasoning/HeF_test.json`, 387 questions

#### FormulaReasoning+
- `data/FormulaReasoning_plus/train.json`, 3608 questions
- `data/FormulaReasoning_plus/HoF_test.json`, 406 questions
- `data/FormulaReasoning_plus/HeF_test.json`, 378 questions

## Requirements
- pytorch
- transformers
- zhipuai
- openai
- dashscope
  
Install numbat tool from [https://github.com/sharkdp/numbat].

## Baselines
#### LLMs
- `GLM` series: baselines/LLMs/GLM/ChatGLM4_api.py
- `GPT` series: baselines/LLMs/GLM/ChatGPT_api.py
- `Qwen` series: baselines/LLMs/GLM/Qwen_api.py
- `DeepSeek` series: baselines/LLMs/deepseek/deepseek_api.py
- `openrouter-api` series: baselines/LLMs/openrouter/openrouter_api.py
- eval: `cd baselines/LLMs/ && python eval_results.py --hof_results {HoF_result_file} --hef_results {HeF_result_file}`

#### DPO
We use MCTS to construct preference data, with the code provided in `baselines/MCTS-PRM`. The code for DPO can be found in `baselines/DPO`.

#### Fine-tuned Small Models
- with calculator: `cd baselines/small_models && bash run_qwen.sh`
- without calculator: `cd baselines/small_models && bash run_qwen_wo_cal.sh`


#### Formula Retriever
- train formula retriever: `cd baselines/RAG/ && bash run.sh`
- eval formula retriever: `cd baselines/RAG/ && python eval.py --model_path outputs_retriever`
