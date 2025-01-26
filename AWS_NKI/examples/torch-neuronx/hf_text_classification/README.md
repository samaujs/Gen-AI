## Fine-tune a “Bert base cased" PyTorch model with AWS Trainium / Inferentia (trn1/inf2 instances) using NeuronSDK.
1. Compile model
2. Train / Fine-tune model

## Parsing the parameters from the command line with following formats and examples to run the code :
```
python3 main.py --task=<TASK> --num_workers=<NUM_WORKERS>
                --model_name=<MODEL_NAME> --task_name=<TASK_NAME>
                --max_seq_length=<MAX_SEQ_LENGTH> --batch_size=<BATCH_SIZE>
                --learning_rate=<LEARNING_RATE> --num_train_epochs=<NUM_TRAIN_EPOCHS>
```

#### Compile model :
```
python3 main.py --task=1 --num_workers=2
                --model_name="bert-base-cased" --task_name="mrpc"
                --max_seq_length=128 --batch_size=8
                --learning_rate=2e-05
```

#### Train / Fine-tune model :
```
python3 main.py --task=1 --num_workers=2
                --model_name="bert-base-cased" --task_name="mrpc"
                --max_seq_length=128 --batch_size=8
                --learning_rate=2e-05 --num_train_epochs=5
```

## References :<br>
>[1] [Get Started with Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/quick-start/index.html)<br>
>[2] [Get Started with Neuron on Ubuntu 22 with Neuron Multi-Framework DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/multiframework/multi-framework-ubuntu22-neuron-dlami.html#setup-ubuntu22-multi-framework-dlami)<br>
