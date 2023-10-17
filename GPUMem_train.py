import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "rte"
model_checkpoint = "bert-base-uncased"
batch_size = 1

from datasets import load_dataset, load_metric

actual_task = "mnli" if task == "mnli-mm" else task
# dataset = load_dataset("glue", actual_task, keep_in_memory= True, cache_dir ='/home/yj/.cache/huggingface/datasets' )
# metric = load_metric('glue', actual_task)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
import torch




def prune_model(model):
    import numpy as np
    n_layers, n_heads=12,12
    head_rank = np.load("./importance/"+task+".npy")
    print("head rank",head_rank)
    head_weight = np.sum(head_rank, axis=1)
    print("head weight",head_weight)
    block_rank = {}
    for i in range(len(head_weight)):
        block_rank[i] = head_weight[i]
    block_rank = sorted(block_rank.items(), key=lambda x: x[1])
    block_rank = [x[0] for x in block_rank]
    print("block rank",block_rank)

    head_mask=torch.ones(n_layers, n_heads)
    head_rank=torch.Tensor(head_rank)
    print(head_mask.shape)
    print(head_rank.shape)
    head_mask[head_rank > 100.0] = 0
    heads_to_prune = dict(
        (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist()) for layer in range(len(head_mask))
    )
    print("head to prune",heads_to_prune)
    # model.prune_heads(heads_to_prune)
    # '''
    for key in heads_to_prune:
        prune_heads = heads_to_prune[key]
        prune_heads = [prune_heads] if type(prune_heads) is not type([]) else prune_heads
        heads_to_prune[key]=prune_heads
        # model.decoder.block[key].layer[0].SelfAttention.prune_heads(prune_heads)
        # model.decoder.block[key].layer[1].EncDecAttention.prune_heads(prune_heads)
    print("head to prune", heads_to_prune)
    model.prune_heads(heads_to_prune)
    return block_rank


class LoRAConfig:
    def __init__(self):
        self.lora_rank = 4
        self.lora_init_scale = 0.01
        self.lora_modules = ".*self|.*SelfAttention|.*EncDecAttention"
        self.lora_layers = "query|key|value|output"
        self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*|.*LayerNorm.*"
        self.lora_scaling_rank = 0


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, scaling_rank, init_scale):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)
            if init_scale < 0:
                self.lora_b = nn.Parameter(torch.randn(linear_layer.out_features, rank) * init_scale)
            else:
                self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale
            )
            if init_scale < 0:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                    + torch.randn(linear_layer.out_features, self.scaling_rank) * init_scale
                )
            else:
                self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features, self.scaling_rank))

    def forward(self, input):
        if self.scaling_rank == 1 and self.rank == 0:
            # parsimonious implementation for ia3 and lora scaling
            if self.multi_lora_a.requires_grad:
                hidden = F.linear((input * self.multi_lora_a.flatten()), self.weight, self.bias)
            else:
                hidden = F.linear(input, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                hidden = hidden * self.multi_lora_b.flatten()
            return hidden
        else:
            # general implementation for lora (adding and scaling)
            weight = self.weight
            if self.scaling_rank:
                weight = weight * torch.matmul(self.multi_lora_b, self.multi_lora_a) / self.scaling_rank
            if self.rank:
                weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
            return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, rank={}, scaling_rank={}".format(
            self.in_features, self.out_features, self.bias is not None, self.rank, self.scaling_rank
        )


def modify_with_lora(transformer, config, block_rank):
    for m_name, module in dict(transformer.named_modules()).items():
        # print(m_name)
        if re.fullmatch(config.lora_modules, m_name):
            # print(m_name)
            # print("encoder.layer"+str(block_rank[0]) )


            # if "encoder.layer."+str(block_rank[0]) in m_name:
            #     lora_rank=8
            # elif "encoder.layer."+str(block_rank[1]) in m_name:
            #     lora_rank = 8
            # elif "encoder.layer." + str(block_rank[2]) in m_name:
            #     lora_rank = 8
            # elif "encoder.layer." + str(block_rank[3]) in m_name:
            #     lora_rank = 8
            # else :
            #     lora_rank = 4

            lora_rank = 4
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    # print(c_name,"%%%%%%%%")
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(layer, lora_rank, config.lora_scaling_rank, config.lora_init_scale),
                    )

    for m_name, module in transformer.named_parameters():
        print(m_name)
        if re.fullmatch(config.trainable_param_names, m_name):
            module.requires_grad = True
        else:
            module.requires_grad = False
    return transformer




MB = 1024.0 * 1024.0
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
torch.cuda.synchronize()
now_mem = torch.cuda.max_memory_allocated() / MB
print("Load original model, mem:", now_mem )

loraconfig = LoRAConfig()

# block_rank =prune_model(model)
block_rank=None
model = modify_with_lora(model, loraconfig, block_rank)
model = model.to('cuda')
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
torch.cuda.synchronize()

now_mem = torch.cuda.max_memory_allocated() / MB
print("Model to device, mem:", now_mem )