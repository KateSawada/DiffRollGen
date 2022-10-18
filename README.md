# Table of Content
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=2 orderedList=false} -->

<!-- code_chunk_output -->

- [Table of Content](#table-of-content)
- [Installation](#installation)
- [Training](#training)
  - [Supervised training](#supervised-training)
  - [Unsupervised pretraining](#unsupervised-pretraining)
- [Sampling](#sampling)
  - [Transcription](#transcription)
  - [Generation](#generation)

<!-- /code_chunk_output -->

# Installation
This repo is developed using `python==3.8.10`, so it is recommended to use `python>=3.8.10`.

To install all dependencies
```
pip install -r requirements.txt
```

# Training

## Supervised training
```
python train_spec_roll.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=0.1 dataset=MAESTRO dataloader.train.num_workers=4 epochs=2500 download=True
```


- `gpus` sets which GPU to use. `gpus=[k]` means `device='cuda:k'`, `gpus=2` means [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP) is used with two GPUs.
- `model.args.kernel_size` sets the kernel size for the ResNet layers in DiffRoll. `model.args.kernel_size=9` performs the best according to our experiments.
- `model.args.spec_dropout` sets the dropout rate ($p$ in the paper)
- `dataset` sets the dataset to be trained on. Can be `MAESTRO` or `MAPS`.
- `dataloader.train.num_workers` sets the number of workers for train loader.
- `download` should be set to `True` if you are running the script for the first time to download and setup the dataset automatically. You can set it to `False` if you already have the dataset downloaded.

The checkpoints and training logs are avaliable at `outputs/YYYY-MM-DD/HH-MM-SS/`. 

To check the progress of training using TensorBoard, you can use the command below
```
tensorboard --logdir='./outputs'
```

## Unsupervised pretraining
### Step 1: Pretraining on MAESTRO using only piano rolls
```
python train_spec_roll.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=1 dataset=MAESTRO dataloader.train.num_workers=4 epochs=2500
```

- `model.args.spec_dropout` sets the dropout rate ($p$ in the paper). When it is set to `1`, it means no spectrograms will be used (all spectrograms dropped to `-1`)
- other arguments are same as [Supervised Training](#supervised-training).

The pretrained checkpoints are avaliable at `outputs/YYYY-MM-DD/HH-MM-SS/ClassifierFreeDiffRoll/version_1/checkpoints`.

After this, you can choose one of the methods (Step [2a](#step-2a:-continue-training-on-maps-using-both-spectrograms-and-piano-rolls), 2b, or 2c) to continue training below.


### Step 2a: Continue training on MAPS using both spectrograms and piano rolls

```
python continue_train_single.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=0 dataset=MAPS dataloader.train.num_workers=4 epochs=2500 pretrained_path='path_to_your_weights' 
```

- `pretrained_path` specifies the location of pretrained weights obtained in [Step 1](#step-1:-pretraining-on-maestro-using-only-piano-rolls)
- other arguments are same as [Supervised Training](#supervised-training).

### Step 2b: Continue training on MAPS using both spectrograms and piano rolls
```
python continue_train_single.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=0 dataset=MAPS dataloader.train.num_workers=4 epochs=2500 pretrained_path='path_to_your_weights' 
```

- `pretrained_path` specifies the location of pretrained weights obtained in [Step 1](#step-1:-pretraining-on-maestro-using-only-piano-rolls)
- other arguments are same as [Supervised Training](#supervised-training).


### Step 2b: Continue training on MAPS using both spectrograms and piano rolls and MAESTRO using only piano rolls 

```
python train_both.py gpus=[0] model.args.kernel_size=9 dataset=Both epochs=10000 model.args.spec_dropout=0
```



# Sampling
## Transcription
```python test.py gpus=[1]```


## Inpainting
```python sampling.py task=inpaintingpython sampling.py task=inpainting```

## Generation
```python generate_music.py```