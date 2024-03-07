import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm


# 트레이닝 클래스를 정의합니다.
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source) ###
        output = output.permute(0, 2, 1)
        # print("debug_output:", output)
        loss = F.cross_entropy(output, targets)
        #print(f"debug_loss: {loss}")
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    # def _run_batch(self, source, targets, train: bool = True) -> float:
    #     with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    #         _, loss = self.model(source)
               
    #     self.optimizer.zero_grad(set_to_none=True)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    #     self.optimizer.step()
        
    #     return loss.item()

    def _run_epoch(self, epoch): # 이거는 아까 그거 그대로
        # print(self.train_data)
        # b_sz = len(next(iter(self.train_data))[0])
        # print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        # self.train_data.sampler.set_epoch(epoch)
        # for source, targets in self.train_data:
        #     source = source.to(self.local_rank)
        #     targets = targets.to(self.local_rank)
        #     self._run_batch(source, targets)
        dataloader = self.train_data  # 여기를 이거로 정의하면 되지 않을까요
        dataloader.sampler.set_epoch(epoch) 
        # print("debugtest: ", dataloader[0])
        #for idx, d in enumerate(dataloader):
        #    print("debug_dataloader: ",idx, d)
        # for iter, (source, targets, att_mask) in enumerate(dataloader):
        for iter, tensor_dict in enumerate(tqdm(dataloader)):
            # step_type = "Train" if train else "Eval"
            source = tensor_dict['input_ids'].to(self.local_rank)
            targets = tensor_dict['labels'].to(self.local_rank)
            #print(f"debug_targets: {targets}")
            batch_loss = self._run_batch(source, targets)
            if iter % 100 == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {iter} | Batch Loss {batch_loss}")


    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
                
"""
# Trainer를 상속받는 customTrainer : CodeplayTrainer
class CodeplayTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # call super class method to get the eval outputs
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        return eval_output
"""