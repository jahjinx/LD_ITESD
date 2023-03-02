<h1 style="font-size:300%;">AI4EduRes'2023: <br />Fine-Tuning RoBERTa for Downstream Tasks</h1>

- [This Repo](#this-repo)
- [Setup](#setup)
- [01 EDA](#01-eda)
- [02 Build Models](#02-build-models)
- [03 Evaluate Models](#03-evaluate-models)
- [04 Build Ensembles](#04-build-ensembles)
- [05 Results](#05-results)
- [🙅 Not In This Repo](#-not-in-this-repo)

<br />

# This Repo
This repo contains code that supports my MSc Research at the University of Leeds, which explores leveraging large language models to produce sufficient inducers for ensembles that outperform single-model systems on downstream linguistic tasks. 

Although the code in this repo may be extended for use with additional models and datasets, given the right preperation, it is largely tailored for this research. Without major adjustments, it may support the range of RoBERTa models hosted on HuggingFace or locally as well as additional sequence classification datasets. However, it will not support non-RoBERTa models or multiple-choice tasks other than HellaSwag and CosmosQA without adjustments.

This repo is structured according to the sequence by which the code should be run, beginning with data prep and EDA.

<br />

# Setup
See `requirements.txt` for a list of dependencies and their versions. Additionally, running `scripts/conda-env-setup.sh` will create a compatible Conda environment for use with the code in this repo.

# 01 EDA
The notebooks contained in the `/01_notebooks_EDA` directory are used to secure, explore, and prepare the individual datasets used in this research for training and testing pipelines. Perpetrations include balancing, splitting, and saving the datasets as HuggingFace DatasetDict objects for easier manipulation in subsequent steps. 

<br />

# 02 Build Models
The `/02_build_models` directory contains notes on the training processs for our control, intermediate, and target models as well as examples detailing how to train those models using `trainer.py` and the command line. 

For example, in order to train one control model for 10 epochs, we run the following commands:

```
python3 trainer.py \
--dataset_path "data/target_iSarcasmEval/itesd_iSarcasmEval_balanced.hf" \
--model_path "roberta-base" \
--data_type "sequence_classification" \
--device "mps" \
--num_labels 2 \
--output_dir "model_saves/control_iSarcasm_01"
```

Most model hyperparameters are kept consistent between models and are therefore assigned default values. They may be changed through the command line via their respective argument flags. Run `python3 trainer.py --help` for a full list of available arguments.

<br />

# 03 Evaluate Models
`/03_evaluate_models` contains a notebook and supporting utilities for evaluating the performance of our control, intermediate, and target models on their respective test sets. The results are saved to the directories specified in the notebook.

<br />

# 04 Build Ensembles
`/04_build_ensembles` walks us through the ensemble building process, which includes constructing the range of ensembles we intend to evaluate, aggregating the predictions of each ensemble's component inducers, and saving the results of each ensemble's evaluation to the file specified in the notebook.

<br />

# 05 Results
The results directory contains the exported results of our evaluations as well as notebooks exploring those results.

<br />

# 🙅 Not In This Repo
Due to size and storage limitations, this repo does not contain the specific models built for this research. It also does not include the source nor preprocessed datasets. While the models are not available here, the `model_saves` directory does contain notes on the training output for each model created. You may also secure the source datasets as well as preprocess those datasets appropriately by following the notebooks in `/01_notebooks_EDA`.