#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : utils.py
@author: jahjinx
@contact : ja.h.jinx@gmail.com
@date  : 2022/12/15 10:00
@version: 1.0
@desc  : 
"""

############## Imports ##############
import random
import logging
import numpy as np

import torch
import platform
from transformers import set_seed

############## Settings ##############
# set logging level
logging.basicConfig(format='%(message)s', level=logging.INFO)

############## Classes/Functions ##############
def platform_check():
    if "arm" in platform.platform():
        print(f"We're Armed: {platform.platform()}")
    else:
        print(f"WARNING! NOT ARMED: {platform.platform()}")
        
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
    
def preprocessing_dyna(input_text, tokenizer, max_length, eval=False):
    """
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
        - input_ids: list of token ids
        - attention_mask: list of indices (0,1) specifying which tokens should considered by the model 
          (return_attention_mask = True).
    """
    # Tokenize
    if eval == True: # for evaluation
        tokenized_examples = tokenizer(input_text['text'],
                                       padding = True,
                                       max_length = max_length,
                                       truncation=True)
    else: # for training  
        tokenized_examples = tokenizer(input_text['text'],
                                       add_special_tokens = True,
                                       max_length = max_length,
                                       truncation=True,
                                       return_attention_mask = True)
    return tokenized_examples

def mc_preprocessing(examples, tokenizer,max_length, eval=False):
    """
    This function determines the input data source (HellaSwag or CosmosQA), merges contexts/questions/answers where
    appropriate, then encodes and tokenizes the data. It structures the output into a dictionary acceptable by the dataloaders.
    """
    # hellaswag uses ending0, ending1, ending2, ending3
    if 'ending0' in examples.keys():
        # designate ending names
        ending_names = ["ending0", "ending1", "ending2", "ending3"]
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        full_context = [[context] * 4 for context in examples["ctx_a"]]
        # Grab all second sentences possible for each context.
        question_headers = examples["ctx_b"]
        # merge question headers with options
        options = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]

    # cosmosqa uses answer0, answer1, answer2, answer3
    elif 'answer0' in examples.keys():
        # designate answer names
        ending_names = ["answer0", "answer1", "answer2", "answer3"]
    
        contexts = examples["context"]
        questions = examples["question"]
    
        # Combine the contexts and answers with the separator tokens </s></s> 
        # We use this method as the tokenizer will not combine three sentences into one input.
        # Also, repeat each sentence four times to go with the four possibilities of answers.
        full_context = [[contexts[i] + "</s></s>" + questions[i]]*4 for i in range(len(contexts))]
        options = [[examples[end][i] for end in ending_names] for i in range(len(contexts))]
    
    else:
        print("ERROR: ending names not found")
    
    # Flatten
    full_context = sum(full_context, [])
    options = sum(options, [])
        
    # Tokenize
    if eval == True: # for evaluation
        tokenized_examples = tokenizer(full_context, 
                                       options, 
                                       padding=True, 
                                       max_length=max_length, 
                                       truncation=True)
    else: # for training  
        tokenized_examples = tokenizer(full_context, 
                                       options,
                                       add_special_tokens = True,
                                       max_length = max_length,
                                       truncation=True,
                                       return_attention_mask = True)
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}       

def construct_input(encoded_dataset_split):
    """
    Takes a DatasetDict split that includes "input_ids", "attention_mask", 
    "label" features and constructs a list of dictionaries for the DataLoaders.
    """
    logging.info('Constructing Input...')
    accepted_keys = ["input_ids", "attention_mask", "label"]
    num_samples = len(encoded_dataset_split)
    
    features = [{k: v for k, v in encoded_dataset_split[i].items() if k in accepted_keys} for i in range(num_samples)]
    
    return features 

def collate(features, tokenizer):
    """
    Data collator that will dynamically pad the inputs via dataloader fn.
    """
    label_name = "label" if "label" in features[0].keys() else "labels"
    labels = [feature.get(label_name) for feature in features]
    labeless_feat = extract_inputs(label_name, features)
    
    # check if features are for multiple choice or sequence classification
    # multiple choice: input_ids is a list of lists (one list for each choice)
    if type(features[0]['input_ids'][0]) == list:
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in labeless_feat]
        flattened_features = sum(flattened_features, [])
    
        batch = pad_batch(flattened_features, tokenizer)
        
        # Un-flatten
        batch = [v.view(batch_size, num_choices, -1) for k, v in batch.items()]
    
    # sequence classification: input_ids is a list of ints
    elif type(features[0]['input_ids'][0]) == int:
        batch = pad_batch(labeless_feat, tokenizer)   
        
        # Un-flatten
        batch = [v for k, v in batch.items()]
    
    else:
        print("Cannot Determine Input Type for Collation")

    # Add back labels
    batch.append(torch.tensor(labels, dtype=torch.int64))
    
    return batch

def extract_inputs(label_name, features):
    """
    Takes dataset label designation as well as dataset input features and 
    returns a list of dictionaries with the input features only.
    This function is primarily used in conjunction with the collate function.
    """
    labeless_feat = []

    # extract input_ids and attention_masks from features
    for feature in features:
        feat_dict = {}
        for k, v in feature.items():
            if k != label_name:
                feat_dict[k] = v
        labeless_feat.append(feat_dict)
        
    return labeless_feat

def pad_batch(adjusted_features, tokenizer):
    """
    Takes input_ids and attention masks which may or may not have been flattened
    according to use and returns a batch of padded input_ids and attention masks
    This function is primarily used in conjunction with the collate function.
    """
    batch = tokenizer.pad(adjusted_features,
                          padding=True,
                          max_length=None,
                          return_tensors="pt")
    return batch

def metric_check(num_labels):
    metric_average = "binary"
    # check num labels for validation metrics
    if num_labels > 2:
        metric_average = "micro"
    
    return metric_average