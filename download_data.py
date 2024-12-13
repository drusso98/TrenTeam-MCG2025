import pandas as pd


# downloading data from huggingface
splits = {'train': 'train.json', 'validation': 'dev.json', 'test': 'test.json'}
df_train = pd.read_json("hf://datasets/LanD-FBK/ML_MTCONAN_KN/" + splits["train"], lines=True)
df_dev = pd.read_json("hf://datasets/LanD-FBK/ML_MTCONAN_KN/" + splits["validation"], lines=True)
df_test = pd.read_json("hf://datasets/LanD-FBK/ML_MTCONAN_KN/" + splits["test"], lines=True)

# saving in /data folder
df_train.T.to_json("data/train.json")
df_dev.T.to_json("data/dev.json")
df_test.T.to_json("data/test.json")

print("\n\n>> datasets saved in /data folder\n\n")
print(" ===== dataset size =====")
print(f"  train: {len(df_train)}")
print(f"  dev: {len(df_dev)}")
print(f"  test: {len(df_test)}")
print(" ========================")