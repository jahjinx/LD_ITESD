import os
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
    model ([`PreTrainedModel`] or `torch.nn.Module`):
        The model to train, evaluate or use for predictions.

    device (`torch.device()`):
        Device on which to put model and data.

    tokenizer ([`PreTrainedTokenizerBase`]):
        The tokenizer used to preprocess the data.

    train_dataloader (`torch.utils.data.DataLoader`):
        PyTorch data loading object to feed model training input data.

    validation_dataloader (`torch.utils.data.DataLoader`):
        PyTorch data loading object to feed model training input data.

    epochs (float):
        Number of epochs the model for which the model with train.

    optimizer (`torch.optim.Optimizer`):

    val_loss_fn (`torch.nn` loss function):

    num_labels (float):
        Number of labels for classification models to predict.
        Defaults to 2.

    output_dir (str):
        Directory to which the model and model checkpoint will save

    save_freq (float):
        Frequency to save the model, by epoch. 
        EX: 1 sets saving to every epoch, 2 to every other epoch.
        If `None`, model will not be saved.
        Defaults to `None`. 

    checkpoint_freq (float):
        Frequency to save the model checkpoints, by epoch. 
        EX: 1 sets checkpointing to every epoch, 2 to every other epoch.
        If `None`, no checkpoints will be saved.
        Defaults to `None`.

    checkpoint_load (str):
        Path to a model's checkpoint.pt file.
        If `None`, no checkpoint will load.
        Defaults to `None`

    phone_number (str):
        Format: `+[COUNTRY_CODE]############`, no spaces, no dashes, no parentheses 
        If not `None` and with valid phone number, will trigger
            an applescript to iMessage epoch updates to given phone number
        If `None`, no message will be sent.
        Defaults to `None`.

    """
    def __init__(self,
                 dataset_path, 
                 data_type,
                 device, 
                 num_labels=2,
                 epochs=10,
                 val_loss_fn=nn.CrossEntropyLoss(), 
                 tokenizer=RobertaTokenizer.from_pretrained("roberta-base"), 
                 output_dir=None, 
                 save_freq=None,
                 checkpoint_freq=None, 
                 checkpoint_load=None,
                 phone_number=None,):

        self.datasets = load_from_disk(dataset_path)
        self.data_type = data_type
        self.device = torch.device(device)
        self.num_labels = num_labels
        self.epochs = epochs 
        self.val_loss_fn = val_loss_fn 
        self.tokenizer = tokenizer
        self.model = self.configure_model() 
        self.optimizer = self.configure_optimizer()
        self.output_dir = output_dir
        self.save_freq = save_freq 
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_load = checkpoint_load
        self.phone_number = phone_number
        
    def configure_optimizer(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), 
                                    lr=params.learning_rate,
                                    weight_decay=params.weight_decay)
        return optimizer

    def configure_model(self):
        if self.data_type == "multiple_choice":
            logging.info('Loading RobertaForMultipleChoice model...')
            model = RobertaForMultipleChoice.from_pretrained('roberta-base',
                                                             num_labels = self.num_labels,
                                                             output_attentions = False,
                                                             output_hidden_states = False,
                                                             )
        elif self.data_type == "sequence_classification":
            # Load the RobertaForSequenceClassification model
            logging.info('Loading RobertaForSequenceClassification model...')
            model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                                     num_labels = self.num_labels,
                                                                     output_attentions = False,
                                                                     output_hidden_states = False,
                                                                     )        
        model.to(self.device)

        return model
    
    def fit(self):
        # check for existing checkpoint
        current_epoch, val_loss = self.load_checkpoint()

        for epoch in range(current_epoch, self.epochs+1):
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
                
                # check num labels for validation metrics
                if self.num_labels > 2:
                    metric_average = "micro"
                else:
                    metric_average = "binary"

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
                if self.phone_number != None:
                    message = f"{self.output_Dir} epoch {epoch}:\nAccuracy: {round(val_acc, 2)} \nF1: {round(val_f1, 2)}"
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

        if self.data_type == "multiple_choice":
            logging.info('Preprocessing for multiple_choice...')
            encoded_datasets = self.datasets.map(mc_preprocessing, 
                                                 batched=True, 
                                                 fn_kwargs={"tokenizer": self.tokenizer,
                                                            'eval': False}) #TODO Paramaterieze eval
            
        elif self.data_type == "sequence_classification":
            logging.info('Preprocessing for sequence_classification...')
            encoded_datasets = self.datasets.map(preprocessing_dyna, 
                                                 batched=True, 
                                                 fn_kwargs={"tokenizer": self.tokenizer}) 

        logging.info(f'Prepping Dataloader for {self.data_type}...')
        features = construct_input(encoded_datasets[split])
        
        dataloader = DataLoader(
                features,
                sampler = RandomSampler(features),
                batch_size = params.batch_size,
                worker_init_fn=seed_worker,
                generator=g,
                # lambda to pass tokenizer to collate_fn via dataloader
                collate_fn=lambda batch: collate(batch, tokenizer=self.tokenizer)
                )
        
        return dataloader
    
    def save_model(self, epoch, model, val_acc=0, val_f1=0):
        if self.save_freq != None and ((epoch)%self.save_freq == 0):
            
            save_name = f'E{str(epoch).zfill(2)}_A{round(val_acc, 2)}_F{round(val_f1, 2)}'

            results_path = os.path.join(self.output_dir, save_name)
            
            try:
                # model save
                model.save_pretrained(results_path)
                self.tokenizer.save_pretrained(results_path)
                logging.info(f'\t * Model @ epoch {epoch} saved to {results_path}')
            
            except Exception:
                logging.info(f'\t ! Model @ epoch {epoch} not saved')
                pass
        
        else:
            logging.info(f"\t ! Save Directory: {self.output_dir}, \
                                Save Frequency: {self.save_freq}, \
                                Epoch: {epoch}")
            
    def save_checkpoint(self, epoch, model, loss,  val_acc=0, val_f1=0):

        if self.checkpoint_freq != None and ((epoch)%self.checkpoint_freq == 0):
            checkpoint_name = f'E{str(epoch).zfill(2)}_A{round(val_acc, 2)}_F{round(val_f1, 2)}'
            results_path = os.path.join(self.output_dir, checkpoint_name, "checkpoint.pt")

            try:
                # model save
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss,
                            }, results_path)
            
                logging.info(f'\t * Model checkpoint saved to {results_path}')
            
            except Exception:
                logging.info(f'\t ! Model checkpoint not saved')
                pass
        
        else:
            logging.info(f"\t ! Checkpoint Directory: {self.output_dir}, \
                                Save Frequency: {self.checkpoint_freq}, \
                                Epoch {epoch}")

    def load_checkpoint(self):
        # load checkpoint if existing
        if self.checkpoint_load != None:
            checkpoint = torch.load(self.checkpoint_load)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            current_epoch = checkpoint['epoch']+1 # increment from last epoch
            val_loss = checkpoint['loss']

        else:
            current_epoch = 1
            val_loss = None

        return current_epoch, val_loss

    def send_message(self, message):
        os.system('osascript scripts/sendMessage.applescript {} "{}"'.format(self.phone_number, message))
        logging.info("\t * Epoch Notification Sent")