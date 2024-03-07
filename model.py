from transformers import AutoConfig, GPT2LMHeadModel
import torch

# 모델 클래스를 정의합니다.
class MidiModel(torch.nn.Module):
    def __init__(self, tokenizer, context_length, n_layer, n_head, n_emb):
        super().__init__()
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            n_positions=context_length,
            n_layer=n_layer,
            n_head=n_head,
            pad_token_id=tokenizer["PAD_None"],
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            n_embd=n_emb,
            # output_hidden_states=True
        )
        self.model = GPT2LMHeadModel(config)
        
    def forward(self, input_ids):
        return self.model(input_ids).logits
        # return self.model(input_ids)
        # return self.model(input_ids).hidden_states[0]