import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "mrpc"
model_checkpoint = "bert-base-uncased"
batch_size = 32

from datasets import load_dataset, load_metric

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
import torch

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


encoded_dataset = dataset.map(preprocess_function, batched=True)


# def prune_model(model):
#     import numpy as np
#     n_layers, n_heads=12,12
#     head_rank = np.load("./importance/rte.npy")
#     print("head rank",head_rank)
#     head_weight = np.sum(head_rank, axis=1)
#     print("head weight",head_weight)
#     block_rank = {}
#     for i in range(len(head_weight)):
#         block_rank[i] = head_weight[i]
#     block_rank = sorted(block_rank.items(), key=lambda x: x[1])
#     block_rank = [x[0] for x in block_rank]
#     print("block rank",block_rank)

#     head_mask=torch.ones(n_layers, n_heads)
#     head_rank=torch.Tensor(head_rank)
#     print(head_mask.shape)
#     print(head_rank.shape)
#     head_mask[head_rank > 100.0] = 0
#     heads_to_prune = dict(
#         (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist()) for layer in range(len(head_mask))
#     )
#     print("head to prune",heads_to_prune)
#     # model.prune_heads(heads_to_prune)
#     # '''
#     for key in heads_to_prune:
#         prune_heads = heads_to_prune[key]
#         prune_heads = [prune_heads] if type(prune_heads) is not type([]) else prune_heads
#         heads_to_prune[key]=prune_heads
#         # model.decoder.block[key].layer[0].SelfAttention.prune_heads(prune_heads)
#         # model.decoder.block[key].layer[1].EncDecAttention.prune_heads(prune_heads)
#     print("head to prune", heads_to_prune)
#     model.prune_heads(heads_to_prune)
#     return block_rank


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


def modify_with_lora(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        # print(m_name)
        if re.fullmatch(config.lora_modules, m_name):

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
        # print(m_name)
        if re.fullmatch(config.trainable_param_names, m_name):
            module.requires_grad = True
        else:
            module.requires_grad = False
    return transformer


loraconfig = LoRAConfig()

# block_rank =prune_model(model)
# model = modify_with_lora(model, loraconfig)
from opendelta import Visualization

model_vis = Visualization(model)
model_vis.structure_graph()
model

metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    acc=metric.compute(predictions=predictions, references=labels)
    print(acc)
    return acc


validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print("Start training!!")

trainer.train()

print("Start Evaluate!!")

trainer.evaluate()

# model.push_to_hub("30-mrpc")
