import json
from datasets import load_dataset

# Load the IFEval dataset from Hugging Face
dataset = load_dataset('google/IFEval', split='train')

# Prepare the data in the required format
data = {"judge_details": []}
for item in dataset:
    data["judge_details"].append({"prompt": item['prompt']})

# Save to JSON file
with open('/data/jiangdazhi/code/research/dInfer/benchmarks/IFEval.json', 'w') as f:
    json.dump(data, f, indent=4)

print("IFEval dataset loaded and saved to IFEval.json")