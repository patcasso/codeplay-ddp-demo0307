import random
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from miditok.pytorch_data import DataCollator
from load_data import load_midi_paths, CodeplayDataset
from torch.utils.data import Dataset, DataLoader

import os

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def load_train_objs(tokenizer):
    midi_paths = ['/home/seohyun/jazz-chunked']
    midi_paths = load_midi_paths(midi_paths)
    print('num of midi files:', len(midi_paths))
    
    dataset_train = CodeplayDataset(
        files_paths=midi_paths,
        min_seq_len=50,
        max_seq_len=1022,
        tokenizer=tokenizer,
    )

    collator = DataCollator(
        tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"], copy_inputs_as_labels=True
    )

    data_loader_train = DataLoader(
        dataset=dataset_train, 
        collate_fn=collator,
        batch_size=16,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset_train)
        )

    return data_loader_train

# def prepare_dataloader(dataset: Dataset, batch_size: int):
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         pin_memory=True,
#         shuffle=False,
#         sampler=DistributedSampler(dataset)
#     )


def split_train_valid(data, valid_ratio=0.1, shuffle=False, seed=42):
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    total_num = len(data)
    num_valid = round(total_num * valid_ratio)
    train_data, valid_data = data[num_valid:], data[:num_valid]
    return train_data, valid_data