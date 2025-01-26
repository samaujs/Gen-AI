#######################################################################################################
# This file defines the method to parse the command line parameter for compiling and training
# - Defines the parameters to be parsed and initialised with the respective default values
# Filename    : parse_nki_args.py
# Created by  : Au Jit Seah
#######################################################################################################
"""
Parsing the parameters from the command line with following formats and examples :

python3 main.py --task=<TASK> --num_workers=<NUM_WORKERS>
                --model_name=<MODEL_NAME> --task_name=<TASK_NAME>
                --max_seq_length=<MAX_SEQ_LENGTH> --batch_size=<BATCH_SIZE>
                --learning_rate=<LEARNING_RATE> --num_train_epochs=<NUM_TRAIN_EPOCHS>

Implements method :
- arg_parse
"""

import argparse

# Set compiling or training parameters
def arg_parse():
    print("Attempt to parse arguments...")
    parser = argparse.ArgumentParser(description='Fine-tune a \"bert-base-cased\" PyTorch model for Text Classification.')

    # Add parsing arguments for compiling or training
    parser.add_argument('--task', dest='task', type=int,
                        help='1: Compile; 2: Train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers.')
    parser.add_argument('--model_name', dest='model_name', type=str,
                        help='Model name.')
    parser.add_argument('--task_name', dest='task_name', type=str,
                        help='Task name.')
    parser.add_argument('--max_seq_length', dest='max_seq_length', type=int,
                        help='Maximum sequence length.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        help='Learning rate.')
    parser.add_argument('--num_train_epochs', dest='num_train_epochs', type=int,
                        help='Number of training epochs.')

    # Set defaults for all program parameters unless provided by user
    parser.set_defaults(task=1,                         # Compile
                        num_workers=2,                  # Number of workers
                        model_name="bert-base-cased",   # Model name for fine-tuning
                        task_name="mrpc",               # Task name
                        max_seq_length=128,             # Maximum sequence length
                        batch_size=8,                   # Batch size
                        learning_rate=2e-05,            # Learning rate

                        num_train_epochs=5)             # Number of training epochs

    return parser.parse_args()
