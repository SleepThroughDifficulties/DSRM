

#不同程度下的uap扰动对于对抗样本和正常样本的影响

import argparse
import json
import os
import random
from pathlib import Path
import model.utils as utils
import textattack
from torch.utils.data import DataLoader
import torch
from textattack.attack_recipes import TextFoolerJin2019
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
parser = argparse.ArgumentParser()
    # settings
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument("--dataset_name", default='glue', type=str)
parser.add_argument("--task_name", default='sst2', type=str)
parser.add_argument('--ckpt_dir', type=Path, default=Path('./saved_models/'))
parser.add_argument('--num_labels', type=int, default=2)
parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
parser.add_argument('--do_train', type=bool, default=True)
parser.add_argument('--do_test', type=bool, default=False)
parser.add_argument('--do_lower_case', type=bool, default=True)
parser.add_argument('--save_models', type=int, default=1)

# adversarial attack
parser.add_argument('--do_attack', type=int, default=0)
parser.add_argument("--num_examples", default=872, type=int)
parser.add_argument('--result_file', type=str, default='attack_result.csv')

# hyper-parameters
parser.add_argument('--max_seq_length', type=int, default=128)
parser.add_argument('--bsz', type=int, default=32)
parser.add_argument('--eval_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
parser.add_argument("--warmup_ratio", default=0.1, type=float,
                    help="Linear warmup over warmup_steps.")  # BERT default
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--bias_correction', default=True)
parser.add_argument('-f', '--force_overwrite', default=True)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
if args.ckpt_dir is not None:
    os.makedirs(args.ckpt_dir, exist_ok=True)
else:
    args.ckpt_dir = '.'
def load_data(tokenizer, args):
    # dataloader
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    # for training and dev
    train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                            subset=args.task_name, split='validation')
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    test_loader = dev_loader


    return train_dataset, train_loader, dev_loader, test_loader


attack_path= Path(os.path.join(args.ckpt_dir, 'UAP_{}_{}-{}_epochs{}_seed{}'
                                       .format(args.model_name, args.dataset_name, args.task_name,
                                            args.epochs, args.seed)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
# tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
# model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
config = AutoConfig.from_pretrained(attack_path)
model = AutoModelForSequenceClassification.from_pretrained(attack_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(attack_path)
model.to(device)
model.eval()
train_dataset, train_loader,dev_loader, test_loader  = load_data(tokenizer, args)
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

attack = TextFoolerJin2019.build(model_wrapper)

from tqdm import tqdm
pbar_train = tqdm(train_loader)
pbar_test = tqdm(test_loader)
with open(os.path.join('results', args.task_name, 'deltas0.json'), 'r') as f1:
    delta0s = json.load(f1)
with open(os.path.join('results', args.task_name, 'deltas1.json'), 'r') as File2:
    delta1s = json.load(File2)

#build adverial example
attack_text_pairs = []
for model_inputs, labels in pbar_test:
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    labels = labels.to(device)
    model.zero_grad()
    word_embedding_layer = model.get_input_embeddings()
    input_ids = model_inputs['input_ids']
    del model_inputs['input_ids']  # new modified
    attention_mask = model_inputs['attention_mask']
    embedding_init = word_embedding_layer(input_ids)
    input_lengths = torch.sum(attention_mask, 1)
    input_mask = attention_mask.to(embedding_init)
    mask_tensor = input_mask.unsqueeze(2)
    repeat_shape = mask_tensor.shape
    i = 0
    texts = []
    model_inputs['inputs_embeds'] = embedding_init
    f_texts = []


    for ids in input_ids:
        attack_text_pair = {}
        ppss = tokenizer.convert_ids_to_tokens(ids)
        text = tokenizer.convert_tokens_to_string(ppss[1:int(input_lengths[i]-1)])
        f_texts.append(text)
        label = labels[i]
        attack_result = attack.attack(text, int(label)).perturbed_text()
        texts.append(attack_result)
        i = i+1

    multi_x = tokenizer.batch_encode_plus(texts, max_length=128, truncation=True, padding=True,
                                          return_tensors='pt')
    multi_x = {k: v.to(device) for k, v in multi_x.items()}
    input_ids_2 = multi_x['input_ids']
    del multi_x['input_ids']  # new modified
    attention_mask_2 = multi_x['attention_mask']
    embedding_init_2 = word_embedding_layer(input_ids_2)
    input_mask_2 = attention_mask_2.to(embedding_init_2)
    mask_tensor_2 = input_mask_2.unsqueeze(2)
    repeat_shape_2 = mask_tensor_2.shape

    for i in range(11):
        delta0 = torch.tensor(random.choice(delta0s)).to(device).view(1, 1, -1) * i / 10
        # delta0 是把预测都转换成0的扰动
        delta1 = torch.tensor(random.choice(delta1s)).to(device).view(1, 1, -1) * i / 10
        delta_train = delta0.repeat(repeat_shape[0], repeat_shape[1], 1) * mask_tensor * labels.view(-1, 1, 1) \
                      + delta1.repeat(repeat_shape[0], repeat_shape[1], 1) * mask_tensor * (1 - labels).view(-1, 1, 1)

        delta_attack = delta0.repeat(repeat_shape_2[0], repeat_shape_2[1], 1) * mask_tensor_2 * (1 - labels).view(-1, 1, 1) \
                       + delta1.repeat(repeat_shape_2[0], repeat_shape_2[1], 1) * mask_tensor_2 * labels.view(-1, 1, 1)

        model_inputs['inputs_embeds'] = delta_train.to(torch.float32) + embedding_init
        multi_x['inputs_embeds'] = delta_attack.to(torch.float32) + embedding_init_2

        logits = model(**model_inputs).logits
        _, preds = logits.max(dim=-1)
        logits_2 = model(**multi_x).logits
        _, preds_2 = logits_2.max(dim=-1)
        is_perturn = []
        for i in range(len(texts)):
            is_perturn.append(texts[i]==f_texts[i])
        print(f'preturn rate now: {i/10} \n labels for train: {preds} \n labels for attack: {preds_2} \n '
              f'labels: {labels} \n  is preturn: {is_perturn}')





    print(1)



