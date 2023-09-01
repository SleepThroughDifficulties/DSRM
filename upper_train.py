"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import json
import time
import logging
import os
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from tqdm import tqdm
import sys
import higher

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import csv
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)

import model.utils as utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--cuda', type=str, default="3")
    parser.add_argument('--date', type=str, default='time is ?')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default='sst2', type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('./saved_models/Upper_models'))
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--save_models', type=int, default=1)

    # adversarial attack
    parser.add_argument('--do_attack', type=bool, default=True)
    parser.add_argument("--num_examples", default=872, type=int)
    parser.add_argument('--result_file', type=str, default='attack_result.csv')

    # hyper-parameters
    parser.add_argument('--max_seq_length', type=int, default=128) # imdb 256, other 128
    parser.add_argument('--bsz', type=int, default=32) # imdb 16, other 32
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10) # hyper-para
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float)  # BERT default
    parser.add_argument('--seed', type=int, default=46) # 46
    parser.add_argument('--bias_correction', default=True)
    parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')

    # Ours hyper-para
    parser.add_argument('--shifting', type=int, default=0, help='是否使用分布偏移的权重训练')
    parser.add_argument('--ad_eval', type=bool, default=False, help='是否使用对抗数据的eval')
    parser.add_argument('--loss_clamp', type=float, default=0.1, help='分布偏移的loss边界')
    parser.add_argument('--modeldir', default="temp", type=str)

    args = parser.parse_args()
    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '.'

    if args.dataset_name == 'imdb':
        args.num_labels = 2
    elif args.dataset_name == 'ag_news':
        args.num_labels = 4
    elif args.dataset_name == 'glue' and args.task_name == 'sst2':
        args.num_labels = 2
    
    

    args.output_dir = Path(os.path.join(args.ckpt_dir, 'Date{}_ds{}_epochs{}_seed{}_meta{}_clamp{}'
            .format(args.date, args.dataset_name, args.epochs, args.seed, args.shifting, args.loss_clamp)))

    print(f"output_dir is {args.output_dir}")

    if not args.output_dir.exists():
        os.makedirs(args.output_dir, exist_ok=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')


    return args


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def adversarial_attack(output_dir, args):
    
    for epoch in range(args.epochs):

        attack_path = Path(str(output_dir) + '/epoch' + str(epoch))
        original_accuracy, accuracy_under_attack, attack_succ = attack_test(attack_path, args)

        out_csv = open(args.result_file, 'a', encoding='utf-8', newline="")
        csv_writer = csv.writer(out_csv)
        csv_writer.writerow([attack_path, original_accuracy, accuracy_under_attack, attack_succ])
        out_csv.close()
    


def attack_test(attack_path, args):

    from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
    from textattack.datasets import HuggingFaceDataset
    from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
    from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper
    from textattack import Attacker
    from textattack import AttackArgs

    # for model
    config = AutoConfig.from_pretrained(attack_path)
    model = AutoModelForSequenceClassification.from_pretrained(attack_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(attack_path)
    model.eval()

    # for dataset
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = TextFoolerJin2019.build(model_wrapper)

    if args.dataset_name in ['imdb', 'ag_news']:
        attack_valid = 'test'
    elif args.task_name == 'mnli':
        attack_valid = 'validation_matched'
    else:
        attack_valid = 'validation'

    dataset = HuggingFaceDataset(args.dataset_name, args.task_name, split=attack_valid)

    # for attack
    attack_args = AttackArgs(num_examples=args.num_examples,
                             disable_stdout=True, random_seed=args.seed)
    attacker = Attacker(attack, dataset, attack_args)
    num_results = 0
    num_successes = 0
    num_failures = 0
    for result in attacker.attack_dataset():
        num_results += 1
        if (
                type(result) == SuccessfulAttackResult
                or type(result) == MaximizedAttackResult
        ):
            num_successes += 1
        if type(result) == FailedAttackResult:
            num_failures += 1

    original_accuracy = (num_successes + num_failures) * 100.0 / num_results
    accuracy_under_attack = num_failures * 100.0 / num_results

    if original_accuracy != 0:
        attack_succ = (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy
    else:
        attack_succ = 0

    return original_accuracy, accuracy_under_attack, attack_succ



def load_data(tokenizer, args):
    # dataloader
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    # for training and dev
    train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)

    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        split_ratio = 0.1
        train_size = round(int(len(train_dataset) * (1 - split_ratio)))
        dev_size = int(len(train_dataset)) - train_size
        # train and dev dataloader
        train_dataset, dev_dataset = torch.utils.data.random_split(train_dataset, [train_size, dev_size])
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator, drop_last=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

        test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    elif args.task_name == 'mnli':
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator, drop_last=True)
        dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='validation_matched')
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
        test_loader = dev_loader
    elif args.dataset_name == 'glue' and args.task_name == 'sst2': 
        split_ratio = 0.1
        train_size = round(int(len(train_dataset) * (1 - split_ratio)))
        dev_size = int(len(train_dataset)) - train_size
        # train and dev dataloader
        train_dataset, dev_dataset = torch.utils.data.random_split(train_dataset, [train_size, dev_size])
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator, drop_last=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

        test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='validation')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    return train_dataset, train_loader, dev_dataset, dev_loader, test_dataset, test_loader


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    avg_loss = utils.ExponentialMovingAverage()
    with torch.no_grad():
        for texts, model_inputs, labels in data_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            logits = model(**model_inputs).logits
            _, preds = logits.max(dim=-1)
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            avg_loss.update(loss.item())
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)
        accuracy = correct / (total + 1e-13)
    return accuracy, avg_loss.get_metric()


def get_eval_batch(args, eval_pairs, tokenizer):    #读取对抗样本
    text_batch = []
    labels = []
    if args.bsz % 2 == 1:
        raise KeyError('batch size需要是个偶数')
    if args.ad_eval: # False
        pairs = random.sample(eval_pairs, int(args.bsz))
        for pair in pairs:
            text = pair['text']
            label = int(pair['label'])
            attack_text = pair['attack_text']
            text_batch.append(text)
            text_batch.append(attack_text)
            labels.append(label)
            labels.append(label)
    else:
        pairs = random.sample(eval_pairs, int(args.bsz*2))
        for pair in pairs:
            text = pair['text']
            label = int(pair['label'])
            text_batch.append(text)
            labels.append(label)
    if len(labels) == len(text_batch):
        batch_data = tokenizer.batch_encode_plus(text_batch, max_length=args.max_seq_length, truncation=True, padding=True,
                                          return_tensors='pt')
        return batch_data, torch.tensor(labels)
    else:
        raise KeyError('长度不匹配了兄弟')


class meta_model(nn.Module):
    def __init__(self, args, config):
        super(meta_model, self).__init__()
        self.args = args
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
        self.weights = nn.Linear(args.bsz, 1, bias=False)
        nn.init.constant_(self.weights.weight, 1)

    def refresh_weights(self):
        nn.init.constant_(self.weights.weight, 1)

    def forward(self, model_inputs, labels, requir_weight=False):
        logits = self.model(**model_inputs).logits  # (1) backward
        losses = F.cross_entropy(logits, labels.squeeze(-1), reduction='none')
        # losses = torch.sum(logits, dim = -1)
        if requir_weight:
            loss = self.weights(losses) / self.args.bsz
            return loss
        else:
            loss = torch.mean(losses)
            return loss

    def train_forward(self, model_inputs, labels):
        logits = self.model(**model_inputs).logits  # (1) backward
        loss = F.cross_entropy(logits, labels.squeeze(-1), reduction='none')
        return loss, logits


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    set_seed(args.seed)

    log_file = os.path.join(args.output_dir, 'INFO.log')
    logger.addHandler(logging.FileHandler(log_file))

    # pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)

    model = meta_model(args, config)
    model.to(device)

    train_dataset, train_loader, dev_dataset, dev_loader, test_dataset, test_loader = load_data(tokenizer, args)


    eval_pairs = []
    for i in dev_dataset:
        # 0: (text, )   1: input_ids...    2: label
        eval_pairs.append({'text': i[0][0], 'label': i[2]})
    
    print(f'Eval pairs length is {len(eval_pairs)}')

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]


    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.adam_epsilon,
        # correct_bias=args.bias_correction
    )

    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    best_dev_epoch, best_dev_accuracy, test_accuracy = 0, 0, 0

    runt = 0

    for epoch in range(args.epochs):
        model.train()
        avg_loss = utils.ExponentialMovingAverage()
        pbar = tqdm(train_loader, mininterval=2, ncols=200)

        for texts, model_inputs, labels in pbar:

            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)

            valid_inputs, valid_labels = get_eval_batch(args, eval_pairs, tokenizer)
            valid_inputs = {k: v.to(device) for k, v in valid_inputs.items()}
            valid_labels = valid_labels.to(device)
            valid_loss = 0

            bt = time.time()

            losses, _ = model.train_forward(model_inputs, labels)

            if args.shifting and torch.mean(losses) < args.loss_clamp:
            # if args.shifting:
                model.refresh_weights()
                with higher.innerloop_ctx(model, optimizer, device=device) as (fmodel, diffopt):
                    fmodel.train()

                    loss = fmodel(model_inputs, labels, requir_weight=True)
                    diffopt.step(loss)

                    valid_loss = fmodel(valid_inputs, valid_labels, requir_weight=False)
                    paras = fmodel.parameters(time=0)
                    for para in paras:
                        pass
                    weight_grads = torch.autograd.grad(valid_loss, para)[-1]
                #distribution shift
                weight_grads = 1e-8 + weight_grads / len(weight_grads)
                lam = (args.loss_clamp - torch.mean(losses)) / torch.matmul(weight_grads.unsqueeze(1), losses)

                #TODO
                w = torch.clamp(lam * weight_grads * args.bsz + 1, -30, 30)

            else:
                w = torch.ones(args.bsz).to(device)

            #TODO
            if epoch in [3, 5, 7, 9]:
                logger.info(f'w = : {w}')


            # losses = torch.sum(logits, dim = -1)
            loss = torch.mean(losses * w.detach())
            loss.backward()

            total_loss = 0
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            runt += (time.time() - bt)

            avg_loss.update(total_loss)

            pbar.set_description(f'epoch: {epoch: d}, '
                                 f'avg_loss: {avg_loss.get_metric(): 0.4f}, '
                                 f'loss: {loss:0.3f}, '
                                 f'valid loss: {valid_loss:0.3f}')

        bert_model = model.model
        if args.save_models:
            s = os.path.join(args.output_dir, 'epoch'+str(epoch))
            if not os.path.exists(s):
                os.makedirs(s, exist_ok=True)
            bert_model.save_pretrained(s)
            tokenizer.save_pretrained(s)
            torch.save(args, os.path.join(s, "training_args.bin"))

        # test after one epoch
        dev_accuracy, dev_loss = evaluate(bert_model, test_loader, device)
        logger.info(f'Epoch: {epoch}, '
                    f'Runtime: {runt: 0.4f}, '
                    f'Loss: {avg_loss.get_metric(): 0.4f}, '
                    f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                    f'Dev_Accuracy: {dev_accuracy}')

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_dev_epoch = epoch
            test_accuracy, test_loss = evaluate(bert_model, test_loader, device)
            logger.info(f'**** Test Accuracy: {test_accuracy}, Test_Loss: {test_loss}')
            if args.save_models:
                bert_model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    logger.info(f'**** Best dev metric: {best_dev_accuracy} in Epoch: {best_dev_epoch}')
    logger.info(f'**** Best Test metric: {test_accuracy} in Epoch: {best_dev_epoch}')

    last_test_accuracy, last_test_loss = evaluate(bert_model, test_loader, device)
    logger.info(f'Last epoch test_accuracy: {last_test_accuracy}, test_loss: {last_test_loss}')


    # attack
    if args.do_attack:
        adversarial_attack(args.output_dir, args)


if __name__ == '__main__':

    args = parse_args()

    print('INFO: ----------------args----------------\n')
    for k in list(vars(args).keys()):
        print(f'INFO: {k}: {vars(args)[k]}\n')
    print('INFO: ----------------args----------------\n')

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
