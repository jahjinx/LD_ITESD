import os
import argparse
import numpy as np

from utils import *

import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader, RandomSampler

import logging
from tqdm import tqdm
from datasets import load_from_disk

from transformers import RobertaTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaForMultipleChoice

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# suppress MPS CPU fallback warning
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

# suppress model warning
from transformers import logging
logging.set_verbosity_error()

# set logging level
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

# clean slates
from datasets import disable_caching
disable_caching()

# set general seeds
set_seeds(1)

# set dataloader generator seed
g = torch.Generator()
g.manual_seed(1)

class Trainer:
    """
    Trainer is a simple training and eval loop for PyTorch, including tdqm 
        and iMessage sending for progress monitoring.
    
    Args
    ---------------
    See argparse and argparse help below.
    """
    def __init__(self,
                 args: argparse.Namespace):
        self.args = args   
        self.param_log = self.log_params()  # log params to ensure consistency
        self.datasets = load_from_disk(self.args.dataset_path)
        self.device = torch.device(self.args.device)
        self.model = self.configure_model() 
        self.optimizer = self.configure_optimizer()        
        self.val_loss_fn=nn.CrossEntropyLoss()         
        self.tokenizer=RobertaTokenizer.from_pretrained(self.args.model_path, use_fast=True) 

        
    def configure_optimizer(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), 
                                    lr=self.args.learning_rate,
                                    weight_decay=self.args.weight_decay)
        return optimizer

    def configure_model(self):
        if self.args.data_type == "multiple_choice":
            logging.info('Loading RobertaForMultipleChoice model...')
            model = RobertaForMultipleChoice.from_pretrained(self.args.model_path,
                                                             num_labels = self.args.num_labels,
                                                             output_attentions = False,
                                                             output_hidden_states = False, 
                                                             local_files_only=self.args.local_files,
                                                             # Ignore for fine-tuning Target from Intermediate with different sized num_labels
                                                             ignore_mismatched_sizes=True,) 
    
        elif self.args.data_type == "sequence_classification":
            logging.info('Loading RobertaForSequenceClassification model...')
            model = RobertaForSequenceClassification.from_pretrained(self.args.model_path,
                                                                     num_labels = self.args.num_labels,
                                                                     output_attentions = False,
                                                                     output_hidden_states = False,
                                                                     local_files_only=self.args.local_files,
                                                                     # Ignore for fine-tuning Target from Intermediate with different sized num_labels
                                                                     ignore_mismatched_sizes=True,)
        model.to(self.device)

        return model
    
    def fit(self):
        # check for existing checkpoint
        current_epoch, val_loss = self.load_checkpoint()
        
        for epoch in range(current_epoch, self.args.epochs+1):
            # ==================== Training ====================
            # Set model to training mode
            self.model.train()
            
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            # tqdm for progress bars
            with tqdm(self.train_dataloader(), unit="batch") as tepoch:
                for step, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    train_output = self.model(b_input_ids, 
                                        token_type_ids = None, 
                                        attention_mask = b_input_mask, 
                                        labels = b_labels)
                    
                    # training_loss = compute_loss(train_output.logits, b_labels) # custom loss

                    # Backward pass
                    train_output.loss.backward()
                    # training_loss.backward() # custom loss

                    self.optimizer.step()
                    # Update tracking variables
                    tr_loss += train_output.loss.item()
                    # tr_loss += training_loss.item() # custom loss
                    nb_tr_examples += b_input_ids.size(0)
                    nb_tr_steps += 1

                # ==================== Validate ====================
                
                # set F1 to binary or micro based on num labels
                metric_average = metric_check(self.args.num_labels)

                val_loss, val_acc, val_f1, val_recall, val_precision = self.validate(self.model, 
                                                                                     self.validation_dataloader(), 
                                                                                     self.device, 
                                                                                     self.val_loss_fn,
                                                                                     metric_average)
                
                # log training information    
                logging.info('\n \t - Train loss: {:.6f}'.format(tr_loss / nb_tr_steps))
                logging.info('\t - Validation Loss: {:.6f}'.format(val_loss))
                logging.info('\t - Validation Accuracy: {:.6f}'.format(val_acc))
                logging.info('\t - Validation F1: {:.6f}'.format(val_f1))
                logging.info('\t - Validation Recall: {:.6f}'.format(val_recall))
                logging.info('\t - Validation Precision: {:.6f} \n'.format(val_precision))
                
                # ==================== Save ====================
                self.save_model(epoch, self.model, val_acc, val_f1)
                self.save_checkpoint(epoch, self.model, val_loss,  val_acc, val_f1)

                # just to make the log prettier
                logging.info('')
                
                # ==================== Notify ====================
                if self.args.phone_number != None:
                    message = f"{self.args.output_Dir} epoch {epoch}:\nAccuracy: {round(val_acc, 2)} \nF1: {round(val_f1, 2)}"
                    self.send_message(message)

    def validate(self, model, val_dl, device, loss_fn, metric_average):
        model.eval()

        val_loss = 0.0
        batch_accuracies = []
        batch_f1s = []
        batch_recalls = []
        batch_precisions = []

        with tqdm(val_dl, unit="batch") as prog:
            for step, batch in enumerate(prog):
                prog.set_description(f"\t Validation {step}")

                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                with torch.no_grad():
                    # Forward pass
                    eval_output = model(b_input_ids, 
                                        token_type_ids = None, 
                                        attention_mask = b_input_mask)
                
                    loss = loss_fn(eval_output.logits, b_labels)
                    val_loss += loss.data.item() * b_input_ids.size(0)

                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate validation metrics
                preds = np.argmax(logits, axis = 1).flatten()
                true_labels = label_ids.flatten()

                # accuracy
                batch_accuracy = accuracy_score(true_labels, preds)
                batch_accuracies.append(batch_accuracy)
                
                # f1
                batch_f1 = f1_score(true_labels, preds, zero_division=0, average=metric_average)
                batch_f1s.append(batch_f1)

                # recall
                batch_recall =recall_score(true_labels, preds, zero_division=0, average=metric_average)
                batch_recalls.append(batch_recall)

                # precision
                batch_precision = precision_score(true_labels, preds, zero_division=0, average=metric_average)
                batch_precisions.append(batch_precision)

        val_loss = val_loss / len(val_dl.dataset)
        validation_accuracy = sum(batch_accuracies)/len(batch_accuracies)
        validation_f1 = sum(batch_f1s)/len(batch_f1s)
        validation_recall = sum(batch_recalls)/len(batch_recalls)
        validation_precision = sum(batch_precisions)/len(batch_precisions)

        return val_loss, validation_accuracy, validation_f1, validation_recall, validation_precision
            
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(split="train")

    def validation_dataloader(self) -> DataLoader:
        return self.get_dataloader(split="validation")

    def get_dataloader(self, split) -> DataLoader:

        if self.args.data_type == "multiple_choice":
            logging.info('Preprocessing for multiple_choice...')
            encoded_datasets = self.datasets.map(mc_preprocessing, 
                                                 batched=True, 
                                                 fn_kwargs={"tokenizer": self.tokenizer,
                                                            "max_length": self.args.max_length,
                                                            'eval': False}) #TODO argparse eval?
            
        elif self.args.data_type == "sequence_classification":
            logging.info('Preprocessing for sequence_classification...')
            encoded_datasets = self.datasets.map(preprocessing_dyna, 
                                                 batched=True, 
                                                 fn_kwargs={"tokenizer": self.tokenizer,
                                                            "max_length": self.args.max_length,
                                                            'eval': False}) 

        logging.info(f'Prepping Dataloader for {self.args.data_type}, {split} split...')
        features = construct_input(encoded_datasets[split])
        dataloader = DataLoader(features,
                                sampler = RandomSampler(features),
                                batch_size = self.args.batch_size,
                                worker_init_fn=seed_worker,
                                generator=g,
                                # lambda to pass tokenizer to collate_fn via dataloader
                                collate_fn=lambda batch: collate(batch, tokenizer=self.tokenizer))        
        return dataloader
    
    def save_model(self, epoch, model, val_acc=0, val_f1=0):
        if self.args.save_freq != None and ((epoch)%self.args.save_freq == 0):
            
            save_name = f'E{str(epoch).zfill(2)}_A{round(val_acc, 2)}_F{round(val_f1, 2)}'

            results_path = os.path.join(self.args.output_dir, save_name)
            
            try:
                # model save
                model.save_pretrained(results_path)
                self.tokenizer.save_pretrained(results_path)
                logging.info(f'\t * Model @ epoch {epoch} saved to {results_path}')
            
            except Exception:
                logging.info(f'\t ! Model @ epoch {epoch} not saved')
                pass
        
        else:
            logging.info(f"\t ! Save Directory: {self.args.output_dir}, \
                                Save Frequency: {self.args.save_freq}, \
                                Epoch: {epoch}")
            
    def save_checkpoint(self, epoch, model, loss,  val_acc=0, val_f1=0):

        if self.args.checkpoint_freq != None and ((epoch)%self.args.checkpoint_freq == 0):
            checkpoint_name = f'E{str(epoch).zfill(2)}_A{round(val_acc, 2)}_F{round(val_f1, 2)}'
            results_path = os.path.join(self.args.output_dir, checkpoint_name, "checkpoint.pt")

            try:
                # model save
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss}, results_path)
            
                logging.info(f'\t * Model checkpoint saved to {results_path}')
            
            except Exception:
                logging.info(f'\t ! Model checkpoint not saved')
                pass
        
        else:
            logging.info(f"\t ! Checkpoint Not Saved this Epoch \n \
                           \tCheckpoint Directory: {self.args.output_dir} \n \
                           \tSave Frequency: {self.args.checkpoint_freq} \n \
                           \tEpoch {epoch}")

    def load_checkpoint(self):
        # load checkpoint if existing
        if self.args.checkpoint_load_path != None:
            checkpoint = torch.load(self.args.checkpoint_load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            current_epoch = checkpoint['epoch']+1 # increment from last epoch
            val_loss = checkpoint['loss']

        else:
            current_epoch = 1
            val_loss = None

        return current_epoch, val_loss

    def log_params(self):
        logging.info(f"""
            Training Dataset: {self.args.dataset_path}
            Number of Labels: {self.args.num_labels}
            Batch Size: {self.args.batch_size}
            Learning Rate: {self.args.learning_rate}
            Weight Decay: {self.args.weight_decay}
            Epochs: {self.args.epochs}
            Output Directory: {self.args.output_dir}
            Save Frequency: {self.args.save_freq}
            Checkpoint Frequency: {self.args.checkpoint_freq}
            Max Length: {self.args.max_length}
            """)
        
    def send_message(self, message):
        os.system('osascript scripts/sendMessage.applescript {} "{}"'.format(self.args.phone_number, message))
        logging.info("\t * Epoch Notification Sent")
        
def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    # data parameters
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to HuggingFace DatasetDict or Similarly Compatible Dataset Object")
    parser.add_argument("--data_type", choices=["sequence_classification", "multiple_choice"], required=False, type=str, default='sequence_classification', help="Type of Task for This Model to Solve")
    # model
    parser.add_argument("--model_path", required=False, type=str, default='roberta-base', help="Path to Model for Fine-Tuning and Intermediate Fine Tuning. Use `roberta-base` for to download from HuggingFace")
    parser.add_argument("--local_files", required=False, type=bool, default=False, help="Set to True if loading a Local Model and Tokenizer")
    parser.add_argument("--checkpoint_load_path", required=False, type=str, default=None, help="Path to a Model's Checkpoint.pt File. If `None`, No Checkpoint Will Load.")
    # model hyperparameters
    parser.add_argument("--device", choices=["cpu", "gpu", "mps"], type=str, default="cpu", help="cpu, gpu, or mps")
    parser.add_argument("--num_labels",  type=int, default=2, help="Number of Labels in Dataset")
    parser.add_argument("--epochs",  type=int, default=10, help="Number of Epochs to Train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size for Dataloader")
    parser.add_argument("--learning_rate", type=float, default=1e-05, help="Learning Rate for Optimizer")
    parser.add_argument("--weight_decay",  type=float, default=0.0, help="Weight Decay for Optimizer")
    parser.add_argument("--max_length", type=int, default=256, help="Length to which we Truncate/Pad Sequences")
    # saving
    parser.add_argument("--output_dir", type=str, help="Path to Save/Checkpoint Model")
    parser.add_argument("--save_freq",  type=int, default=1, help="How Often to Save Model (By Epoch)")
    parser.add_argument("--checkpoint_freq", type=int, default=2, help="How Often to Save Checkpoint (Float, By Epoch, 'None' if No Checkpointing is Required")
    parser.add_argument("--phone_number", type=str, default=None, help="`+[COUNTRY_CODE]############`, no spaces, no dashes, no parentheses ")
    return parser

def train(args):
    trainer = Trainer(args)
    trainer.fit()
    
def main():
    parser = get_parser()
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()