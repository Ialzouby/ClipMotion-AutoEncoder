from datasets import load_dataset

ds = load_dataset("TeoGchx/HumanML3D", split="train")
print(ds[0])
