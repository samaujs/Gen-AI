#######################################################################################################
# This file defines the main method for fine-tuning the model
# - Calls the respective methods to compile and fine-tune the model
# Filename    : main.py
# Created by  : Au Jit Seah
#######################################################################################################
"""
Implements the main method to call the methods to fine-tune the model

Implements methods :
- compile_model
- train_model
"""

# Import common libraries
import time
import subprocess

# Parsing input parameters
import parse_nki_args

def compile_model(num_workers: int = 2, model_name: str = "bert-base-cased", task_name: str = "mrpc",
                  max_seq_length: int = 128, batch_size: int = 8, learning_rate: float = 2e-05):
    """
    params:
    - num_workers (int)      : Number of workers
    - model_name (str)       : Model name for fine-tuning
    - task_name (str)        : Task name
    - max_seq_length (int)   : Maximum sequence length
    - batch_size (int)       : Batch size
    - learning_rate (float)  : Learning rate

    Compile the model
    """

    env_var_options = "XLA_USE_BF16=1 NEURON_CC_FLAGS=\"--model-type=transformer\""
    model_base_name = model_name
    max_train_samples = 128

    # Subdirectory
    exec_file = 'transformers/examples/pytorch/text-classification/run_glue.py'

    COMPILE_CMD = f"""{env_var_options} neuron_parallel_compile \
                      torchrun --nproc_per_node={num_workers} \
                      {exec_file} \
                      --model_name_or_path {model_name} \
                      --task_name {task_name} \
                      --do_train \
                      --max_seq_length {max_seq_length} \
                      --per_device_train_batch_size {batch_size} \
                      --learning_rate {learning_rate} \
                      --max_train_samples {max_train_samples} \
                      --overwrite_output_dir \
                      --output_dir {model_base_name}-{task_name}-{batch_size}bs"""

    print(f'Execute command:\n{COMPILE_CMD}')
    if subprocess.check_call(COMPILE_CMD,shell=True):
        print("There was an error with the compilation command")
    else:
        print("Compilation Success!!!")


def train_model(num_workers: int = 2, model_name: str = "bert-base-cased", task_name: str = "mrpc",
                max_seq_length: int = 128, batch_size: int = 8, learning_rate: float = 2e-05,
                num_train_epochs: int = 5):
    """
    params:
    - num_workers (int)      : Number of workers
    - model_name (str)       : Model name for fine-tuning
    - task_name (str)        : Task name
    - max_seq_length (int)   : Maximum sequence length
    - batch_size (int)       : Batch size
    - learning_rate (float)  : Learning rate
    - num_train_epochs (int) : Number of training epochs

    Train the model
    """

    env_var_options = "XLA_USE_BF16=1 NEURON_CC_FLAGS=\"--model-type=transformer\""
    model_base_name = model_name
    max_train_samples = 128

    # Subdirectory
    exec_file = 'transformers/examples/pytorch/text-classification/run_glue.py'

    TRAIN_CMD = f"""{env_var_options} torchrun --nproc_per_node={num_workers} \
                    {exec_file} \
                    --model_name_or_path {model_name} \
                    --task_name {task_name} \
                    --do_train \
                    --do_eval \
                    --max_seq_length {max_seq_length} \
                    --per_device_train_batch_size {batch_size} \
                    --learning_rate {learning_rate} \
                    --num_train_epochs {num_train_epochs} \
                    --overwrite_output_dir \
                    --output_dir {model_base_name}-{task_name}-{num_workers}w-{batch_size}bs"""

    print(f'Execute command:\n{TRAIN_CMD}')
    if subprocess.check_call(TRAIN_CMD,shell=True):
        print("There was an error with the fine-tune command")
    else:
        print("Fine-tune Success!!!")


# Start of Program from command line
def main():
    # Parsing defaults for all program parameters unless provided by user
    prog_args = parse_nki_args.arg_parse()

    if prog_args.task == 1:
        print(f"** Compile model : {prog_args.model_name} **")
        print("---------------------------------------------")
        print()

        # Compile model
        compile_model(num_workers=prog_args.num_workers, model_name=prog_args.model_name, task_name=prog_args.task_name,
                      max_seq_length=prog_args.max_seq_length, batch_size=prog_args.batch_size, learning_rate=prog_args.learning_rate)

    elif prog_args.task == 2:
        print(f"** Train model : {prog_args.model_name} **")
        print("-------------------------------------------")
        print()

        # Train model
        train_model(num_workers=prog_args.num_workers, model_name=prog_args.model_name, task_name=prog_args.task_name,
                    max_seq_length=prog_args.max_seq_length, batch_size=prog_args.batch_size, learning_rate=prog_args.learning_rate,
                    num_train_epochs=prog_args.num_train_epochs)

    else:
          print("Supported task option is only; 1: Compile model or 2: Train model")



# Main Program
if __name__ == "__main__":
    main()
