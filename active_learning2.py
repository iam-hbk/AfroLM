# Adapted for convenience from https://github.com/castorini/afriberta/blob/6cacc453f3a99a6f902174e8e7f8dd6184c1794f/main.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import pipeline
from absl import flags

from source.trainer import TrainingManager
from source.utils import load_config

experiment_name = "active_learning_lm"

EXPERIMENT_PATH = "experiments_500k"
EXPERIMENT_CONFIG_NAME = "config.yml"

if not os.path.exists(EXPERIMENT_PATH):
    os.mkdir(EXPERIMENT_PATH)

experiment_path = os.path.join(EXPERIMENT_PATH, experiment_name)

if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

experiment_config_path = os.path.join(experiment_path, EXPERIMENT_CONFIG_NAME)

flags.DEFINE_string("config_path", "models_configurations/large.yml", "Config file path")

config = load_config("models_configurations/large.yml")

langs = ['amh', 'hau', 'lug', 'luo', 'pcm', 'sna', 'tsn', 'wol', 'ewe', 'bam', 'bbj', 'mos', 'zul', 'lin', 'nya', 'twi',
         'fon', 'ibo', 'kin', 'swa', 'xho', 'yor', 'oro']

dataset = 'dataset/{}_mono.tsv'

def save_list(lines, filename):
    try:
        with open(filename, 'w', encoding="utf-8") as file:
            data = '\n'.join(str(line).strip() for line in lines)
            file.write(data)
    except IOError as e:
        print(f"An error occurred when writing to the file {filename}: {e}")

def main():
    active_learning_steps = 3
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/eval', exist_ok=True)
    os.makedirs('data/txt', exist_ok=True)

    for step in range(1, active_learning_steps + 1):
        print(f'Active Learning Step: {step}')
        all_evals = []
        for lang in langs:
            try:
                current_dataset = pd.read_csv(dataset.format(lang), sep='\t')
                if 'input' not in current_dataset.columns:
                    raise ValueError(f"The 'input' column is missing in the dataset for language {lang}.")
                current_dataset = current_dataset.sample(frac=1, random_state=1234)
                train, test = train_test_split(current_dataset, test_size=0.2, random_state=1234)
                all_evals.extend(test['input'].tolist())
                save_list(train['input'].tolist(), f'data/train/train.{lang}')
                save_list(test['input'].tolist(), f'data/eval/eval.{lang}')
            except FileNotFoundError:
                print(f"The dataset file for language {lang} was not found.")
            except pd.errors.EmptyDataError:
                print(f"No data found for language {lang}.")
            except Exception as e:
                print(f"An error occurred while processing the dataset for language {lang}: {e}")

        save_list(all_evals, 'data/eval/all_eval.txt')
        trainer = TrainingManager(config, experiment_path, step)
        trainer.train()

if __name__ == '__main__':
    main()
