import yaml
import numpy as np
from cs336_basics.tokenizer import BPETokenizer

config = yaml.safe_load(open("config/train_tinystories.yaml", "r"))

tokenizer = BPETokenizer(
    vocab = config["tokenizer"]["vocab_file"],
    merges= config["tokenizer"]["merges_file"],
    special_tokens=config["tokenizer"]["special_tokens"]
)

tokenized_data = tokenizer.parallel_tokenize_txt(config["dataset"]["train_file"], 4)
tokenized_data = np.array(tokenized_data, dtype=np.short)

# with open(config["dataset"]["tokenized_train_file"], "rb") as f:
#     tokenized_data = np.load(f, allow_pickle=True)
#     tokenized_data = np.array(tokenized_data, dtype=np.short)

with open(config["dataset"]["tokenized_train_file"], "wb") as f:
    np.save(f, tokenized_data, allow_pickle=True)

print(f"Tokenization completed with {len(tokenized_data)} tokens. Saved to {config['dataset']['tokenized_train_file']}")
