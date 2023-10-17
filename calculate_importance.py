

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "stsb"
model_checkpoint = "bert-base-uncased"
batch_size = 1

from datasets import load_dataset, load_metric
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
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


import torch

n_layers, n_heads = 12,12
head_importance = torch.zeros(n_layers, n_heads)
head_mask = torch.ones(n_layers, n_heads)
head_mask.requires_grad_(requires_grad=True)
tot_tokens=0.0
for i, batch in enumerate(encoded_dataset["train"]):
    if i ==3:
        break
    if i != 0:
        head_mask.grad.data.zero_()
    labels = torch.LongTensor([batch['label']])
    if task == 'stsb':
        labels = torch.FloatTensor([batch['label']]) 
    input_ids = torch.LongTensor([batch['input_ids']])
    attention_mask = torch.LongTensor([batch['attention_mask']])
    # print(type(input_ids))
    # print(labels,input_ids,attention_mask)
    # print(input_ids.shape)
    outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            # #encoder_outputs=None,
            head_mask=head_mask,
            # decoder_head_mask=decoder_head_mask.cuda()
        )
    loss=outputs[0]
    loss.backward()
    head_importance += head_mask.grad.abs().detach()
    tot_tokens += attention_mask.float().detach().sum().data
    # print(outputs)

head_importance /= tot_tokens

exponent = 2
norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())


head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long)
head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
    head_importance.numel()
)
head_ranks = head_ranks.view_as(head_importance)

print(head_ranks)


import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(head_ranks, linewidths = 0.05, cmap="YlGnBu")
# ax1.set_title(task)
# ax1.set_xlabel('num_hidden_layers')
# ax1.set_xticklabels([]) #设置x轴图例为空值
# ax1.set_ylabel('num_attention_heads')
plt.savefig("importance/"+task+".png",)
import numpy as np
np.save("importance/"+task+".npy",head_ranks.cpu().detach().numpy())
