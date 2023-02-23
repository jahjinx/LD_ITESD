# import torch
from transformers import RobertaTokenizer

# dataset_path: str          = None
# # --------- Model Parameters --------- 
# num_labels: float          = 2 # number of output labels for current model

# # --------- Training Parameters --------- 
batch_size: float          = 16 # Recommended batch size: {16, 32}. See: https://arxiv.org/pdf/1907.11692.pdf
learning_rate: float       = 1e-05 # Recommended Learning Rates {1e−5, 2e−5, 3e−5}. See: https://arxiv.org/pdf/1907.11692.pdf
weight_decay: float        = 0 # RoBERTa Paper does not mention hyperparameter search for weight decay

# epochs: float              = 10 # Recommended number of epochs: 10. See: https://arxiv.org/pdf/1907.11692.pdf


# phone_number: str          = "" # w/+Country Code, no spaces, dashes, or parentheses

# # --------- Tokenizer Parameters --------- 
# tokenizer                  = RobertaTokenizer.from_pretrained("roberta-base")  # used in utils, fix
max_length: float          = 256 # length of tokenized phrases allowed, 512 max for RoBERTa