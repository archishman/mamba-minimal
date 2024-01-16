from model import Mamba, ModelArgs
from transformers import AutoTokenizer

# One of:
#     'state-spaces/mamba-2.8b-slimpj'
#     'state-spaces/mamba-2.8b'
#     'state-spaces/mamba-1.4b'
#     'state-spaces/mamba-790m'
#     'state-spaces/mamba-370m'
#     'state-spaces/mamba-130m'
pretrained_model_name = 'state-spaces/mamba-370m'

model = Mamba.from_pretrained(pretrained_model_name)
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
model.eval()
model.to("mps")
import torch
import torch.nn.functional as F


def generate(model,
             tokenizer,
             prompt: str,
             n_tokens_to_gen: int = 10,
             sample: bool = False,
             top_k: int = None):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt').to("mps")
    input_ids = input_ids.input_ids
    
    for token_n in range(n_tokens_to_gen):
        print(torch.mps.synchronize(), 0, token_n)
        with torch.no_grad():
            if token_n == 12:
                pass #breakpoint()
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input)[:, -1]
        print(torch.mps.synchronize(), 1)
        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape
        print(torch.mps.synchronize(), 2)
        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)
        print(torch.mps.synchronize(), 3)
        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]
        print(torch.mps.synchronize(), 4)
        input_ids = torch.cat([input_ids, next_indices], dim=1)
        print(torch.mps.synchronize(), 4.5)
    print(5)
    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]
    
    return output_completions

print(generate(model, tokenizer, 'Mamba is the'))