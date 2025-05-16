from datasets import load_dataset

def load_data(data_dir, split):
    dataset = load_dataset(data_dir, split=split)
    examples = list(dataset)
    
    # 如果示例中没有 "idx" 字段，则添加 "idx" 字段
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]
    
    # 处理 gsm8k 数据集的特定情况
    if "gsm8k" in data_dir:
        examples = [
            {
                "idx": example["idx"],
                "question": example["question"],
                "ground_truth": example["answer"].split("#### ")[-1].strip()
            } for example in examples
        ]
    elif "math" in data_dir:
        examples = [
            {
                "idx": example["idx"],
                "question": example["problem"],
                "ground_truth": example["answer"]
            } for example in examples
        ]
    return examples

if __name__ == "__main__":
    split = "train"
    data = load_data("/home/xli/skw/Qwen2.5-Math/evaluation/data/math", split)
    print(data[0])
