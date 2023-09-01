"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
import os
from pathlib import Path
import random
import numpy as np
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch

import csv
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)

from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.attack_recipes import (PWWSRen2019,
                                       BAEGarg2019,
                                       TextBuggerLi2018
                                       )

from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper
from textattack import Attack
from textattack import Attacker
from textattack import AttackArgs

from textattack.datasets import HuggingFaceDataset
from textattack.constraints.pre_transformation import InputColumnModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))



def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default='sst2', type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('./saved_models/bakUpper_models'))
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--attack_method', type=str, default='bertattack') # bertattack, textfooler, pwws

    # adversarial attack
    parser.add_argument("--num_examples", default=872, type=int) # sst2(glue) 872, imdb and agnews 1000
    parser.add_argument('--result_file', type=str, default='attack_result.csv')

    # parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--cuda', default="0", type=str)
    parser.add_argument('--modeldir', default="Date09031647_ds_glue-sst2_epochs10_seed46_metaTrue_clamp0.1", type=str)

    parser.add_argument("--neighbour_vocab_size", default=10, type=int)
    parser.add_argument("--modify_ratio", default=0.15, type=float)
    parser.add_argument("--sentence_similarity", default=0.85, type=float)

    args = parser.parse_args()
    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '.'

    if args.attack_method == 'bertattack':
        args.neighbour_vocab_size = 50
        args.modify_ratio = 0.9
        args.sentence_similarity = 0.2

    return args


def build_default_attacker(args, model):
    attacker = None
    if args.attack_method == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    elif args.attack_method == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args.attack_method == 'bertattack':
        attacker = BAEGarg2019.build(model)
    elif args.attack_method == 'pwws':
        attacker = PWWSRen2019.build(model)
    else:
        print("Not implement attck!")
        exit(41)
    input_column_modification0= InputColumnModification(["sentence1", "sentence2"], {"sentence1"})
    input_column_modification1 = InputColumnModification(["sentence", "question"], {"sentence"})
    attacker.pre_transformation_constraints.append(input_column_modification0)
    attacker.pre_transformation_constraints.append(input_column_modification1)
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)


def build_weak_attacker(args, model):
    attacker = None
    if args.attack_method == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    elif args.attack_method == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args.attack_method == 'bertattack':
        attacker = BAEGarg2019.build(model)
    elif args.attack_method == 'pwws':
        attacker = PWWSRen2019.build(model)
    else:
        print("Not implement attck!")
        exit(41)

    if args.attack_method in ['bertattack']:
        attacker.transformation = WordSwapEmbedding(max_candidates=args.neighbour_vocab_size)
        for constraint in attacker.constraints:
            if isinstance(constraint, WordEmbeddingDistance):
                attacker.constraints.remove(constraint)
            if isinstance(constraint, UniversalSentenceEncoder):
                attacker.constraints.remove(constraint)

    # attacker.constraints.append(MaxWordsPerturbed(max_percent=args.modify_ratio))
    use_constraint = UniversalSentenceEncoder(
        threshold=args.sentence_similarity,
        metric="cosine",
        compare_against_original=True,
        window_size=15,
        skip_text_shorter_than_window=False,
    )
    attacker.constraints.append(use_constraint)
    input_column_modification0= InputColumnModification(["sentence1", "sentence2"], {"sentence1"})
    input_column_modification1 = InputColumnModification(["sentence", "question"], {"sentence"})
    attacker.pre_transformation_constraints.append(input_column_modification0)
    attacker.pre_transformation_constraints.append(input_column_modification1)
    attacker.goal_function = UntargetedClassification(model)
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)



def adversarial_attack(output_dir, args):

    for epoch in range(8, args.epochs): # Ablustr 09/09

        attack_path = Path(str(output_dir) + '/epoch' + str(epoch))
        original_accuracy, accuracy_under_attack, attack_succ = attack_test(attack_path, args)

        out_csv = open(args.result_file, 'a', encoding='utf-8', newline="")
        csv_writer = csv.writer(out_csv)
        csv_writer.writerow([attack_path, original_accuracy, accuracy_under_attack, attack_succ])
        out_csv.close()
    pass


def attack_test(attack_path, args):

    # for model
    config = AutoConfig.from_pretrained(attack_path)
    model = AutoModelForSequenceClassification.from_pretrained(attack_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(attack_path)
    model.eval()

    # for dataset
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    if args.attack_method == "bertattack":
        attack = build_weak_attacker(args, model_wrapper)
    else:
        attack = build_default_attacker(args, model_wrapper)

    if args.dataset_name in ['imdb', 'ag_news']:
        attack_valid = 'test'
    elif args.task_name == 'mnli':
        attack_valid = 'validation_matched'
    else:
        attack_valid = 'validation'

    if args.dataset_name == 'glue':
        dataset = HuggingFaceDataset(args.dataset_name, args.task_name, split=attack_valid)
    else:
        dataset = HuggingFaceDataset(args.dataset_name, split=attack_valid)

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


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def main(args):
    set_seed(args.seed)

    model_dir = Path(os.path.join(args.ckpt_dir, args.modeldir))
    if not model_dir.exists():
        logger.info(f'no such model_dir')
        return 0

    adversarial_attack(model_dir, args)


if __name__ == '__main__':

    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    main(args)
