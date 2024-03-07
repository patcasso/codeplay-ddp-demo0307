import torch
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from tokenizer import get_custom_tokenizer


from trainer import Trainer
from utils import ddp_setup, load_train_objs#, prepare_dataloader
from model import MidiModel

MAX_SEQ_LEN, BATCH_SIZE = 1024, 8 # 8마디 용


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    
    # tokenizer와 모델 설정
    tokenizer = get_custom_tokenizer()
    context_length = MAX_SEQ_LEN # context length는 자유롭게 바꿔보며 실험해봐도 좋을 듯 합니다.

    # Change this based on size of the data
    n_layer=6
    n_head=4
    n_emb=1024

    model = MidiModel(tokenizer, context_length, n_layer, n_head, n_emb)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_data = load_train_objs(tokenizer)
    # train_data = prepare_dataloader(train_data, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    main(args.save_every, args.total_epochs, args.batch_size)


# trainer_config(training_args) 는 어디에??
# evaluation step 은 어디에??