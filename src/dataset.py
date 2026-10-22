from datasets import load_dataset

ds = load_dataset("OpAI-Bench1/OpAI-Bench", "default")

train = ds['train']

keys = train[0].keys()

for key in keys:
    print(f'{key}: {train[0][key]}')
