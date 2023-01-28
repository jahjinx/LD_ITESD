import torch
import random
import platform

import numpy as np
import params
from transformers import set_seed

#TODO combine preprocessing and collate functions
#TODO look up best way to do so. ex: one function that checks data type and returns appropriate collate function?
#TODO or, do we add everything to the same function and use if statements to check data type?

def seed_worker(worker_id):
    "seeding function for dataloaders"
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seeds(seed):
    "set seeds for reproducibility"
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

def platform_check():
    if "arm" in platform.platform():
        print(f"We're Armed: {platform.platform()}")
    else:
        print(f"WARNING! NOT ARMED: {platform.platform()}")

def preprocessing(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = params.max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

def preprocessing_dyna(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = params.max_length,
                        truncation=True,
                        return_attention_mask = True,
                   )
  
# the following function is for preprocessing multiple-choice data
def hella_preprocessing(examples, eval=False):
    # designate ending names
    ending_names = ["ending0", "ending1", "ending2", "ending3"]

    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["ctx_a"]]

    # Grab all second sentences possible for each context.
    question_headers = examples["ctx_b"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    # Tokenize
    if eval == True:
        tokenized_examples = params.tokenizer(first_sentences, second_sentences, padding=True, truncation=True, max_length=512)
    else:   
        tokenized_examples = params.tokenizer(first_sentences, second_sentences, truncation=True)
        
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

# the following function is for preprocessing multiple-choice data
def cosmos_preprocessing(examples, eval=False):
    # designate answer names
    ending_names = ["answer0", "answer1", "answer2", "answer3"]
    
    contexts = examples["context"]
    questions = examples["question"]
    
    # Combine the contexts and answers with the separator tokens </s></s> 
    # We use this method as the tokenizer will not combine three sentences into one input.
    # Also, repeat each sentence four times to go with the four possibilities of answers.
    cont_quest = [[contexts[i] + "</s></s>" + questions[i]]*4 for i in range(len(contexts))]
    answers = [[examples[end][i] for end in ending_names] for i in range(len(contexts))]

    # Flatten everything
    cont_quest = sum(cont_quest, [])
    answers = sum(answers, [])
    
    # Tokenize
    if eval == True:
        tokenized_examples = params.tokenizer(cont_quest, answers, padding=True, truncation=True, max_length=512)
    else:   
        tokenized_examples = params.tokenizer(cont_quest, answers, truncation=True)
        
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

def encode_text(text_list):
  """
  takes a list of string sequences and encode them, outputting two lists:
  - a list of token ids
  - a list of attention masks
  
  encoded sequences are truncated to params.max_length but not padded
  
  string sequences may be passed from dataframe as `df.text.values`
  """
  
  token_ids = []
  attention_masks = []

  # encode training text
  for sample in text_list:
    encoding_dict = preprocessing_dyna(sample, params.tokenizer)
    # parse encoding dict to lists
    token_ids.append(encoding_dict['input_ids']) 
    attention_masks.append(encoding_dict['attention_mask'])
    
  return token_ids, attention_masks

def seq_class_collate(features):
  """
  Data collator that will dynamically pad the inputs for sequence classification data via dataloader fn.
  """
  label_name = "label" if "label" in features[0].keys() else "labels"

  labels = [feature.get(label_name) for feature in features]

  labeless_feat = []

  # extract input_ids and attention_masks from features
  for feature in features:
      feat_dict = {}
      for k, v in feature.items():
          if k != label_name:
              feat_dict[k] = v
      labeless_feat.append(feat_dict)

  # batch_size = len(features)

  batch = params.tokenizer.pad(
      labeless_feat,
      padding=True,
      max_length=None,
      return_tensors="pt",
  )
    
  # Un-flatten
  batch = [v for k, v in batch.items()]

  # Add back labels
  batch.append(torch.tensor(labels, dtype=torch.int64))
  return batch

def mc_collate(features):
    """
    Data collator that will dynamically pad the inputs for multiple choice data.
    """

    label_name = "label" if "label" in features[0].keys() else "labels"

    labels = [feature.get(label_name) for feature in features]

    labeless_feat = []
    for feature in features:
        feat_dict = {}
        for k, v in feature.items():
            if k != label_name:
                feat_dict[k] = v
        labeless_feat.append(feat_dict)

    batch_size = len(features)
    num_choices = len(features[0]["input_ids"])
    flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in labeless_feat]
    flattened_features = sum(flattened_features, [])
    
    batch = params.tokenizer.pad(
        flattened_features,
        padding=True,
        max_length=None,
        return_tensors="pt",
    )
    
    # Un-flatten
    batch = [v.view(batch_size, num_choices, -1) for k, v in batch.items()]

    # Add back labels
    batch.append(torch.tensor(labels, dtype=torch.int64))
    return batch

def output_parameters():
    print(f"""
      Training Dataset: {params.dataset_path}
      Number of Labels: {params.num_labels}
      Batch Size: {params.batch_size}
      Learning Rate: {params.learning_rate}
      Weight Decay: {params.weight_decay}
      Epochs: {params.epochs}
      Output Directory: {params.output_dir}
      Save Frequency: {params.save_freq}
      Checkpoint Frequency: {params.checkpoint_freq}
      Max Length: {params.max_length}
      """)