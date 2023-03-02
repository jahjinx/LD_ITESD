import os
from utils import *
import pandas as pd
from tqdm import tqdm

from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaForMultipleChoice
from transformers import TextClassificationPipeline
from sklearn.metrics import accuracy_score, f1_score

# suppress MPS CPU fallback warning
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

# clean slates
from datasets import disable_caching
disable_caching()

def parse_model_dir(top_level_dir):
    """
    Given the top-level-directory containing each epoch of our saved model,
    this function returns all model subdirectory paths for iterative testing.

    Ex: returns array-like
    ['model_saves/iSarcasm_control_01/E01_A0.61_F0.4', 'model_saves/iSarcasm_control_01/E02_A0.83_F0.82']
    """

    models = []
    for root, dirs, files in os.walk(top_level_dir, topdown=False):
        for name in dirs:
            models.append(os.path.join(root, name))
    models.sort()
    return models


def evaluate_model(test_dataframe, model_list, num_labels, results_csv_path):
    """
    1. Takes Test Data
        - test dataframe must include ['text'] and ['label'] fields
    2. Makes predictions from test_dataframe['text']
    3. Compares predictions to test_dataframe['label]
    4. Appends scores, predictions, and model info to csv stored at results_csv_path
        - results_csv_path must have format:
        - {'model_name': [], 'model_epoch': [], 'test_accuracy': [], 'test_f1': [], 'predictions':[]}
    """
    
    # put on device to increase inference speed
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # start testing loop
    for path in model_list:
        PATH = path

        path_elements = PATH.split('/')
        model_name = path_elements[1]
        model_epoch = path_elements[2]

        print(f"Model: {PATH}")
        model = RobertaForSequenceClassification.from_pretrained(PATH, local_files_only=True)
        tokenizer = RobertaTokenizer.from_pretrained(PATH, local_files_only=True)
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

        pipe = TextClassificationPipeline(model=model, 
                                          tokenizer=tokenizer, 
                                          top_k=2, 
                                          device=device)

        test_input = test_dataframe['text'].tolist()
        test_output = []

        model.to(device)
        
        # run tests and append to output
        with tqdm(test_input, unit="test") as prog:
            for step, test in enumerate(prog):
                prog.set_description(f"\tTest {step+1}")
                test_output.append(pipe(test, **tokenizer_kwargs)[0])

        # parse predictions to new list
        predictions = []
        for i in test_output:
            # formatted as LABEL_0 or LABEL_1
            # Remove LABEL_ and cast as int
            prediction = i[0]['label'].replace("LABEL_","")
            predictions.append(int(prediction))

        # set F1 to binary or micro based on num labels
        metric_average = metric_check(num_labels)
        
        print(f"\t- Metric Average: {metric_average}")
            
        f1 = f1_score(test_dataframe['label'], predictions, average=metric_average)
        acc = accuracy_score(test_dataframe['label'], predictions)

        export_results(results_csv_path, model_name, model_epoch, acc, f1, predictions)

        print(f"\t- Accuracy: {acc}")
        print(f"\t- F1: {f1}\n")

def evaluate_mc_model(test_data, model_list, num_labels, results_csv_path):
    """
    1. Takes Test Data
        - test dataframe must include ['text'] and ['label'] fields
    2. Makes predictions from test_data['text']
    3. Compares predictions to test_data['label]
    4. Appends scores, predictions, and model info to csv stored at results_csv_path
        - results_csv_path must have format:
        - {'model_name': [], 'model_epoch': [], 'test_accuracy': [], 'test_f1': [], 'predictions':[]}
    """

    # put on device to increase inference speed
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # start testing loop
    for path in model_list:
        PATH = path

        path_elements = PATH.split('/')
        model_name = path_elements[1]
        model_epoch = path_elements[2]

        print(f"Model: {PATH}")
        model = RobertaForMultipleChoice.from_pretrained(PATH, local_files_only=True)
        tokenizer = RobertaTokenizer.from_pretrained(PATH, local_files_only=True)
        
        encoded_dataset = test_data['test'].map(mc_preprocessing, batched=True, fn_kwargs={"tokenizer": tokenizer,
                                                                                           "eval": True})
        
        test_number_samples = len(encoded_dataset)
        accepted_keys = ["input_ids", "attention_mask", "label"]
        test_features = [{k: v for k, v in encoded_dataset[i].items() if k in accepted_keys} for i in range(test_number_samples)]
        true_labels = [k['label'] for k in test_features]
        model.to(device)

        predictions = []
        
        # run tests and append to output
        with tqdm(test_features, unit="test") as prog:
            for step, test in enumerate(prog):
                prog.set_description(f"\tTest {step+1}")
                
                labels = torch.tensor(0).unsqueeze(0)
                test_input = {"input_ids": torch.IntTensor(test['input_ids']), "attention_mask": torch.IntTensor(test['attention_mask'])}
                outputs = model(**{k: v.unsqueeze(0).to(device) for k, v in test_input.items()}, labels=labels.to(device))
                logits = outputs.logits
                predicted_class = logits.argmax().item()
                predictions.append(int(predicted_class))

        # check num labels for validation metrics
        metric_average = metric_check(num_labels)
            
        f1 = f1_score(true_labels, predictions, average=metric_average)
        acc = accuracy_score(true_labels, predictions)

        export_results(results_csv_path, model_name, model_epoch, acc, f1, predictions)

        print(f"\t- Accuracy: {acc}")
        print(f"\t- F1: {f1}\n")     
        
def export_results(results_csv_path, model_name, model_epoch, acc, f1, predictions):
        results_to_save = {'model_name': [model_name], 'model_epoch': [model_epoch], 'test_accuracy': [acc], 'test_f1': [f1], 'predictions': [predictions]}
        results_to_save_df = pd.DataFrame(data=results_to_save)
        results_to_save_df.to_csv(results_csv_path, mode='a', header=False, index=False)