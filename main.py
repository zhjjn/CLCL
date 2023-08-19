import os
import sys
import pickle
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from tqdm import tqdm, trange
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Subset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from utils import Config, Logger, make_log_dir
from modeling import (
    AutoModelForSequenceClassification,
    CTSModelForSequenceClassification,
    CTSModelForSequenceClassification_LF,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification_SPV,
    AutoModelForSequenceClassification_MIP,
    AutoModelForSequenceClassification_SPV_MIP,
    CTSModelForSequenceClassification_SPV_MIP,
    CTSModelForSequenceClassification_SPV_MIP_LF,
)
from run_classifier_dataset_utils import processors, output_modes, compute_metrics
from data_loader import load_train_data, load_train_data_cts, load_train_data_kf, load_test_data

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"

def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def main():
    # read configs
    config = Config(main_conf_path="./")

    # apply system arguments if exist
    argv = sys.argv[1:]
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = " ".join(sys.argv[1:]).split(" ")
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip("-")
            cmd_arg[arg_name] = arg_value
        config.update_params(cmd_arg)

    args = config
    print(args.__dict__)

    # logger
    if "saves" in args.bert_model:
        log_dir = args.bert_model
        logger = Logger(log_dir)
        config = Config(main_conf_path=log_dir)
        old_args = copy.deepcopy(args)
        args.__dict__.update(config.__dict__)

        args.bert_model = old_args.bert_model
        args.do_train = old_args.do_train
        args.data_dir = old_args.data_dir
        args.task_name = old_args.task_name

        # apply system arguments if exist
        argv = sys.argv[1:]
        if len(argv) > 0:
            cmd_arg = OrderedDict()
            argvs = " ".join(sys.argv[1:]).split(" ")
            for i in range(0, len(argvs), 2):
                arg_name, arg_value = argvs[i], argvs[i + 1]
                arg_name = arg_name.strip("-")
                cmd_arg[arg_name] = arg_value
            config.update_params(cmd_arg)
    else:
        if not os.path.exists("saves"):
            os.mkdir("saves")
        log_dir = make_log_dir(os.path.join("saves", args.bert_model))
        logger = Logger(log_dir)
        config.save(log_dir)
    args.log_dir = log_dir

    # set CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # get dataset and processor
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    # build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = load_pretrained_model(args)

    ########### Training ###########

    # VUA18 / VUA20 for bagging 
    if args.do_train and args.task_name == "vua" and args.num_bagging:
        train_data, gkf = load_train_data_kf(args, logger, processor, task_name, label_list, tokenizer, output_mode)

        for fold, (train_idx, valid_idx) in enumerate(tqdm(gkf, desc="bagging...")):
            if fold != args.bagging_index:
                continue

            print(f"bagging_index = {args.bagging_index}")

            # Load data
            temp_train_data = TensorDataset(*train_data[train_idx])
            train_sampler = RandomSampler(temp_train_data)
            train_dataloader = DataLoader(temp_train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            # Reset Model
            model = load_pretrained_model(args)
            model, best_result = run_train(args, logger, model, train_dataloader, processor, task_name, label_list, tokenizer, output_mode)

            # Test
            all_guids, eval_dataloader = load_test_data(args, logger, processor, task_name, label_list, tokenizer, output_mode)
            preds = run_eval(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=True)
            with open(os.path.join(args.data_dir, f"seed{args.seed}_preds_{fold}.p"), "wb") as f:
                pickle.dump(preds, f)

            # If train data is VUA20, the model needs to be tested on VUAverb as well.
            # You can just adjust the names of data_dir in conditions below for your own data directories.
            if "VUA20" in args.data_dir:
                # Verb
                args.data_dir = "data/VUAverb"
                all_guids, eval_dataloader = load_test_data(args, logger, processor, task_name, label_list, tokenizer, output_mode)
                preds = run_eval(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=True)
                with open(os.path.join(args.data_dir, f"seed{args.seed}_preds_{fold}.p"), "wb") as f:
                    pickle.dump(preds, f)

            logger.info(f"Saved to {logger.log_dir}")
        return                
    
    # VUA18 / VUA20
    if args.do_train and args.task_name == "vua" and args.cts and not args.cl:
        train_data, train_dataloader, _ = load_train_data_cts(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train_cts(
            args,
            logger,
            model,
            train_data,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )
    # TroFi / MOH-X (K-fold)
    elif args.do_train and args.task_name == "trofi" and args.cts and not args.cl:
        k_result = []
        for k in tqdm(range(args.kfold), desc="K-fold"):
            model = load_pretrained_model(args)
            train_data, train_dataloader, _ = load_train_data_cts(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            model, best_result = run_train_cts(
                args,
                logger,
                model,
                train_data,
                train_dataloader,
                processor,
                task_name,
                label_list,
                tokenizer,
                output_mode,
                k,
            )
            k_result.append(best_result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")
    # MAGPIE / SemEval / VNC
    elif args.do_train and args.task_name in ['magpie', 'semeval', 'vnc', 'combine', 'transfer'] and args.cts and not args.cl:
        train_data, train_dataloader, _ = load_train_data_cts(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train_cts(
            args,
            logger,
            model,
            train_data,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )
    
    # VUA18 / VUA20
    if args.do_train and args.task_name == "vua" and not args.cts and args.cl:
        train_data, train_dataloader, train_examples = load_train_data_cts(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train_cl(
            args,
            logger,
            model,
            train_data,
            train_examples,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )
    # TroFi / MOH-X (K-fold)
    elif args.do_train and args.task_name == "trofi" and not args.cts and args.cl:
        k_result = []
        for k in tqdm(range(args.kfold), desc="K-fold"):
            model = load_pretrained_model(args)
            train_data, train_dataloader, train_examples = load_train_data_cts(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            model, best_result = run_train_cl(
                args,
                logger,
                model,
                train_data,
                train_examples,
                train_dataloader,
                processor,
                task_name,
                label_list,
                tokenizer,
                output_mode,
                k,
            )
            k_result.append(best_result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")
    # MAGPIE / SemEval / VNC
    elif args.do_train and args.task_name in ['magpie', 'semeval', 'vnc', 'combine', 'transfer'] and not args.cts and args.cl:
        train_data, train_dataloader, train_examples = load_train_data_cts(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train_cl(
            args,
            logger,
            model,
            train_data,
            train_examples,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )
        
        
    # VUA18 / VUA20
    if args.do_train and args.task_name == "vua" and args.cts and args.cl:
        train_data, train_dataloader, train_examples = load_train_data_cts(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train_cts_cl(
            args,
            logger,
            model,
            train_data,
            train_examples,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )
    # TroFi / MOH-X (K-fold)
    elif args.do_train and args.task_name == "trofi" and args.cts and args.cl:
        k_result = []
        for k in tqdm(range(args.kfold), desc="K-fold"):
            model = load_pretrained_model(args)
            train_data, train_dataloader, train_examples = load_train_data_cts(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            model, best_result = run_train_cts_cl(
                args,
                logger,
                model,
                train_data,
                train_examples,
                train_dataloader,
                processor,
                task_name,
                label_list,
                tokenizer,
                output_mode,
                k,
            )
            k_result.append(best_result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")
    # MAGPIE / SemEval / VNC
    elif args.do_train and args.task_name in ['magpie', 'semeval', 'vnc', 'combine', 'transfer'] and args.cts and args.cl:
        train_data, train_dataloader, train_examples = load_train_data_cts(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train_cts_cl(
            args,
            logger,
            model,
            train_data,
            train_examples,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )

    # VUA18 / VUA20
    if args.do_train and args.task_name == "vua" and not args.cts and not args.cl:
        train_dataloader = load_train_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train(
            args,
            logger,
            model,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )
    # TroFi / MOH-X (K-fold)
    elif args.do_train and args.task_name == "trofi" and not args.cts and not args.cl:
        k_result = []
        for k in tqdm(range(args.kfold), desc="K-fold"):
            model = load_pretrained_model(args)
            train_dataloader = load_train_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            model, best_result = run_train(
                args,
                logger,
                model,
                train_dataloader,
                processor,
                task_name,
                label_list,
                tokenizer,
                output_mode,
                k,
            )
            k_result.append(best_result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")
            
    # MAGPIE / SemEval / VNC
    elif args.do_train and args.task_name in ['magpie', 'semeval', 'vnc', 'combine', 'transfer'] and not args.cts and not args.cl:
        train_data, train_dataloader, _ = load_train_data_cts(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train(
            args,
            logger,
            model,
            train_data,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )

    # Load trained model
    if "saves" in args.bert_model:
        model = load_trained_model(args, model, tokenizer)

    ########### Inference ###########
    # VUA18 / VUA20
    if (args.do_eval or args.do_test) and task_name == "vua" and args.cts:
        # if test data is genre or POS tag data
        if ("genre" in args.data_dir) or ("pos" in args.data_dir):
            if "genre" in args.data_dir:
                targets = ["acad", "conv", "fict", "news"]
            elif "pos" in args.data_dir:
                targets = ["adj", "adv", "noun", "verb"]
            orig_data_dir = args.data_dir
            for idx, target in tqdm(enumerate(targets)):
                logger.info(f"====================== Evaluating {target} =====================")
                args.data_dir = os.path.join(orig_data_dir, target)
                all_guids, eval_dataloader = load_test_data(
                    args, logger, processor, task_name, label_list, tokenizer, output_mode
                )
                run_eval_cts(args, logger, model, eval_dataloader, all_guids, task_name)
        else:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode
            )
            run_eval_cts(args, logger, model, eval_dataloader, all_guids, task_name)
    '''
    # TroFi / MOH-X (K-fold)
    elif (args.do_eval or args.do_test) and args.task_name == "trofi" and args.cts:
        logger.info(f"***** Evaluating with {args.data_dir}")
        k_result = []
        for k in tqdm(range(10), desc="K-fold"):
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval_cts(args, logger, model, eval_dataloader, all_guids, task_name)
            k_result.append(result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")
    '''
            
    if (args.do_eval or args.do_test) and task_name == "vua" and not args.cts:
        # if test data is genre or POS tag data
        if ("genre" in args.data_dir) or ("pos" in args.data_dir):
            if "genre" in args.data_dir:
                targets = ["acad", "conv", "fict", "news"]
            elif "pos" in args.data_dir:
                targets = ["adj", "adv", "noun", "verb"]
            orig_data_dir = args.data_dir
            for idx, target in tqdm(enumerate(targets)):
                logger.info(f"====================== Evaluating {target} =====================")
                args.data_dir = os.path.join(orig_data_dir, target)
                all_guids, eval_dataloader = load_test_data(
                    args, logger, processor, task_name, label_list, tokenizer, output_mode
                )
                run_eval(args, logger, model, eval_dataloader, all_guids, task_name)
        else:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode
            )
            run_eval(args, logger, model, eval_dataloader, all_guids, task_name)
    '''
    # TroFi / MOH-X (K-fold)
    elif (args.do_eval or args.do_test) and args.task_name == "trofi" and not args.cts:
        logger.info(f"***** Evaluating with {args.data_dir}")
        k_result = []
        for k in tqdm(range(10), desc="K-fold"):
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name)
            k_result.append(result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")
    logger.info(f"Saved to {logger.log_dir}")
    '''


def run_train(
    args,
    logger,
    model,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_mode,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch

    # Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_schedule != False or args.lr_schedule.lower() != "none":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = { num_train_optimization_steps}")

    # Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0
        step = 0
        for batch in tqdm(train_dataloader, desc="Iteration", position=0, leave=True):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT"]:
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    label_ids,
                    input_ids_2,
                    input_mask_2,
                    segment_ids_2,
                ) = batch
            else:
                if len(batch) == 4:
                    input_ids, input_mask, segment_ids, label_ids = batch
                else:
                    input_ids, input_mask, segment_ids, label_ids, input_ids_pos, input_mask_pos, segment_ids_pos, label_ids_pos, input_ids_neg, input_mask_neg, segment_ids_neg, label_ids_neg = batch

            # compute loss values
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                logits, _ = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits, _ = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

            # average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule != False or args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()
            step += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        # evaluate
        if args.do_eval:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name)

            # update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    save_model(args, model, tokenizer)
            if args.task_name in ["vua", "magpie", "semeval", "vnc", "transfer", "combine"]:
                save_model(args, model, tokenizer)

    logger.info(f"-----Best Result-----")
    for key in sorted(max_result.keys()):
        logger.info(f"  {key} = {str(max_result[key])}")

    return model, max_result


def run_train_cts(
    args,
    logger,
    model,
    train_data,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_mode,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch

    # Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_schedule.lower() == "warmup_linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )
    elif args.lr_schedule.lower() == "warmup_cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
            num_cycles = (int(args.num_train_epoch) - int(args.warmup_epoch)) / 2,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = { num_train_optimization_steps}")

    # Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    cos = torch.nn.CosineSimilarity(dim=1)
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0
        step = 0
        for batch in tqdm(train_dataloader, desc="Iteration", position=0, leave=True):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT"]:
                input_ids_org, input_mask_org, segment_ids_org, input_ids_2_org, input_mask_2_org, segment_ids_2_org, label_ids_org, input_ids_pos, input_mask_pos, segment_ids_pos, input_ids_2_pos, input_mask_2_pos, segment_ids_2_pos, label_ids_pos, input_ids_neg, input_mask_neg, segment_ids_neg, input_ids_2_neg, input_mask_2_neg, segment_ids_2_neg, label_ids_neg = batch
            else:
                input_ids_org, input_mask_org, segment_ids_org, label_ids_org, input_ids_pos, input_mask_pos, segment_ids_pos, label_ids_pos, input_ids_neg, input_mask_neg, segment_ids_neg, label_ids_neg = batch
            

            # compute loss values
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                logits_org, org_r = model(
                    input_ids_org,
                    target_mask=(segment_ids_org == 1),
                    token_type_ids=segment_ids_org,
                    attention_mask=input_mask_org,
                )
                logits_pos, pos_r = model(
                    input_ids_pos,
                    target_mask=(segment_ids_pos == 1),
                    token_type_ids=segment_ids_pos,
                    attention_mask=input_mask_pos,
                )
                logits_neg, neg_r = model(
                    input_ids_neg,
                    target_mask=(segment_ids_neg == 1),
                    token_type_ids=segment_ids_neg,
                    attention_mask=input_mask_neg,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits_org.view(-1, args.num_labels), label_ids_org.view(-1))
                pos_cos = cos(org_r, pos_r).reshape(org_r.shape[0], 1)
                neg_cos = cos(org_r, neg_r).reshape(org_r.shape[0], 1)
                v = torch.cat((pos_cos, neg_cos), dim = 1)
                exp = torch.exp(v)
                # we only need to take the log of the positive value over the sum of exp. 
                cts_loss = -torch.log(exp[:, 0]/torch.sum(exp, dim = 1)).sum()
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits_org, org_r = model(
                    input_ids_org,
                    input_ids_2_org,
                    target_mask=(segment_ids_org == 1),
                    target_mask_2 = segment_ids_2_org,
                    attention_mask_2=input_mask_2_org,
                    token_type_ids=segment_ids_org,
                    attention_mask=input_mask_org,
                )
                logits_pos, pos_r = model(
                    input_ids_pos,
                    input_ids_2_pos,
                    target_mask=(segment_ids_pos == 1),
                    target_mask_2 = segment_ids_2_pos,
                    attention_mask_2 = input_mask_2_pos,
                    token_type_ids=segment_ids_pos,
                    attention_mask=input_mask_pos,
                )
                logits_neg, neg_r = model(
                    input_ids_neg,
                    input_ids_2_neg,
                    target_mask=(segment_ids_neg == 1),
                    target_mask_2 = segment_ids_2_neg,
                    attention_mask_2 = input_mask_2_neg,
                    token_type_ids=segment_ids_neg,
                    attention_mask=input_mask_neg,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits_org.view(-1, args.num_labels), label_ids_org.view(-1))
                pos_cos = cos(org_r, pos_r).reshape(org_r.shape[0], 1)
                neg_cos = cos(org_r, neg_r).reshape(org_r.shape[0], 1)
                v = torch.cat((pos_cos, neg_cos), dim = 1)
                exp = torch.exp(v)
                # we only need to take the log of the positive value over the sum of exp. 
                cts_loss = -torch.log(exp[:, 0]/torch.sum(exp, dim = 1)).sum()
            
            if args.r_drop:
                kl_loss = compute_kl_loss(logits_org, logits_pos)

            # average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()
                cts_loss = cts_loss.mean()
                if args.r_drop:
                    kl_loss = kl_loss.mean()
                
            loss +=  args.cts_loss_weight * cts_loss
            if args.r_drop:
                loss += args.r_drop_weight * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()
            step += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        # evaluate
        if args.do_eval:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval_cts(args, logger, model, eval_dataloader, all_guids, task_name)

            # update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    save_model(args, model, tokenizer)
                if args.task_name in ["vua", "magpie", "semeval", "vnc", "transfer", "combine"]:
                    save_model(args, model, tokenizer)

    logger.info(f"-----Best Result-----")
    for key in sorted(max_result.keys()):
        logger.info(f"  {key} = {str(max_result[key])}")

    return model, max_result

def run_train_cl(
    args,
    logger,
    model,
    train_data,
    train_examples,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_mode,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch

    # Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_schedule.lower() == "warmup_linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )
    elif args.lr_schedule.lower() == "warmup_cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
            num_cycles = (int(args.num_train_epoch) - int(args.warmup_epoch)) / 2,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = { num_train_optimization_steps}")

    # Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    cl_indices = []
    train_data_guids = [train_example.guid for train_example in train_examples]
    cos = torch.nn.CosineSimilarity(dim=1)
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        difficulties_ = evaluate_difficulty_cts(args, model, train_data)
        #print(difficulties)

        difficulties_ = [[i, d, guid] for i, (d, guid) in enumerate(zip(difficulties_, train_data_guids))]
        difficulties_df = pd.DataFrame(difficulties_, columns = ['ID', 'Difficulty', 'GUID'])
        difficulties = [[i,0] for i in range(len(difficulties_))]
        guid_list = list(set(difficulties_df['GUID'].tolist()))

        for guid in tqdm(guid_list , desc="Difficulties", position=0, leave=True):
            current_examples = difficulties_df.groupby("GUID").get_group(guid)
            target_examples_diff = current_examples['Difficulty'].tolist()
            ids = current_examples['ID'].tolist()
            average_diff = np.mean(target_examples_diff)
            for i in ids:
                difficulties[i][1] = average_diff
        #for i, d in enumerate(tqdm(list(difficulties_df.values), desc="Difficulties", position=0, leave=True)):
        #    label = train_examples[i].label
        #    guid = train_examples[i].guid
        #    target_examples_diff = difficulties_df.groupby("GUID").get_group(guid)['Difficulty'].tolist()
        #    average_diff = np.mean(target_examples_diff)
        #    difficulties.append([i,average_diff])

        print(len(difficulties))
        difficulties = sorted(difficulties, key=lambda x:x[1])
        cl_indices = [d[0] for d in difficulties]
        train_dataset_ordered = Subset(train_data, indices=cl_indices)
        train_sampler = SequentialSampler(train_dataset_ordered)
        train_dataloader = DataLoader(
            train_dataset_ordered,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
        )
        model.train()
        tr_loss = 0
        step = 0
        for batch in tqdm(train_dataloader, desc="Iteration", position=0, leave=True):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT"]:
                input_ids_org, input_mask_org, segment_ids_org, input_ids_2_org, input_mask_2_org, segment_ids_2_org, label_ids_org, input_ids_pos, input_mask_pos, segment_ids_pos, input_ids_2_pos, input_mask_2_pos, segment_ids_2_pos, label_ids_pos, input_ids_neg, input_mask_neg, segment_ids_neg, input_ids_2_neg, input_mask_2_neg, segment_ids_2_neg, label_ids_neg = batch
            else:
                input_ids_org, input_mask_org, segment_ids_org, label_ids_org, input_ids_pos, input_mask_pos, segment_ids_pos, label_ids_pos, input_ids_neg, input_mask_neg, segment_ids_neg, label_ids_neg = batch
            

            # compute loss values
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                logits_org, org_r = model(
                    input_ids_org,
                    target_mask=(segment_ids_org == 1),
                    token_type_ids=segment_ids_org,
                    attention_mask=input_mask_org,
                )
      
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits_org.view(-1, args.num_labels), label_ids_org.view(-1))
                
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits_org, org_r = model(
                    input_ids_org,
                    input_ids_2_org,
                    target_mask=(segment_ids_org == 1),
                    target_mask_2 = segment_ids_2_org,
                    attention_mask_2=input_mask_2_org,
                    token_type_ids=segment_ids_org,
                    attention_mask=input_mask_org,
                )
                
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits_org.view(-1, args.num_labels), label_ids_org.view(-1))
                
            
            if args.r_drop:
                kl_loss = compute_kl_loss(logits_org, logits_pos)

            # average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()
                
                if args.r_drop:
                    kl_loss = kl_loss.mean()
                
            
            if args.r_drop:
                loss += args.r_drop_weight * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()
            step += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        # evaluate
        if args.do_eval:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval_cts(args, logger, model, eval_dataloader, all_guids, task_name)

            # update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    save_model(args, model, tokenizer)
                if args.task_name in ["vua", "magpie", "semeval", "vnc", "transfer", "combine"]:
                    save_model(args, model, tokenizer)

    logger.info(f"-----Best Result-----")
    for key in sorted(max_result.keys()):
        logger.info(f"  {key} = {str(max_result[key])}")

    return model, max_result

def run_train_cts_cl(
    args,
    logger,
    model,
    train_data,
    train_examples,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_mode,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch

    # Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_schedule.lower() == "warmup_linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )
    elif args.lr_schedule.lower() == "warmup_cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
            num_cycles = (int(args.num_train_epoch) - int(args.warmup_epoch)) / 2,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = { num_train_optimization_steps}")

    # Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    cl_indices = []
    train_data_guids = [train_example.guid for train_example in train_examples]
    cos = torch.nn.CosineSimilarity(dim=1)
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        difficulties_ = evaluate_difficulty_cts(args, model, train_data)
        #print(difficulties)
        difficulties_ = [[i, d, guid] for i, (d, guid) in enumerate(zip(difficulties_, train_data_guids))]
        difficulties_df = pd.DataFrame(difficulties_, columns = ['ID', 'Difficulty', 'GUID'])
        difficulties = [[i,0] for i in range(len(difficulties_))]
        guid_list = list(set(difficulties_df['GUID'].tolist()))

        for guid in tqdm(guid_list , desc="Difficulties", position=0, leave=True):
            current_examples = difficulties_df.groupby("GUID").get_group(guid)
            target_examples_diff = current_examples['Difficulty'].tolist()
            ids = current_examples['ID'].tolist()
            average_diff = np.mean(target_examples_diff)
            for i in ids:
                difficulties[i][1] = average_diff
        #for i, d in enumerate(tqdm(list(difficulties_df.values), desc="Difficulties", position=0, leave=True)):
        #    label = train_examples[i].label
        #    guid = train_examples[i].guid
        #    target_examples_diff = difficulties_df.groupby("GUID").get_group(guid)['Difficulty'].tolist()
        #    average_diff = np.mean(target_examples_diff)
        #    difficulties.append([i,average_diff])

        print(len(difficulties))
        difficulties = sorted(difficulties, key=lambda x:x[1])
        cl_indices = [d[0] for d in difficulties]
        train_dataset_ordered = Subset(train_data, indices=cl_indices)
        train_sampler = SequentialSampler(train_dataset_ordered)
        train_dataloader = DataLoader(
            train_dataset_ordered,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
        )
        model.train()
        tr_loss = 0
        step = 0
        for batch in tqdm(train_dataloader, desc="Iteration", position=0, leave=True):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT"]:
                input_ids_org, input_mask_org, segment_ids_org, input_ids_2_org, input_mask_2_org, segment_ids_2_org, label_ids_org, input_ids_pos, input_mask_pos, segment_ids_pos, input_ids_2_pos, input_mask_2_pos, segment_ids_2_pos, label_ids_pos, input_ids_neg, input_mask_neg, segment_ids_neg, input_ids_2_neg, input_mask_2_neg, segment_ids_2_neg, label_ids_neg = batch
            else:
                input_ids_org, input_mask_org, segment_ids_org, label_ids_org, input_ids_pos, input_mask_pos, segment_ids_pos, label_ids_pos, input_ids_neg, input_mask_neg, segment_ids_neg, label_ids_neg = batch
            

            # compute loss values
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                logits_org, org_r = model(
                    input_ids_org,
                    target_mask=(segment_ids_org == 1),
                    token_type_ids=segment_ids_org,
                    attention_mask=input_mask_org,
                )
                logits_pos, pos_r = model(
                    input_ids_pos,
                    target_mask=(segment_ids_pos == 1),
                    token_type_ids=segment_ids_pos,
                    attention_mask=input_mask_pos,
                )
                logits_neg, neg_r = model(
                    input_ids_neg,
                    target_mask=(segment_ids_neg == 1),
                    token_type_ids=segment_ids_neg,
                    attention_mask=input_mask_neg,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits_org.view(-1, args.num_labels), label_ids_org.view(-1))
                pos_cos = cos(org_r, pos_r).reshape(org_r.shape[0], 1)
                neg_cos = cos(org_r, neg_r).reshape(org_r.shape[0], 1)
                v = torch.cat((pos_cos, neg_cos), dim = 1)
                exp = torch.exp(v)
                # we only need to take the log of the positive value over the sum of exp. 
                cts_loss = -torch.log(exp[:, 0]/torch.sum(exp, dim = 1)).sum()
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits_org, org_r = model(
                    input_ids_org,
                    input_ids_2_org,
                    target_mask=(segment_ids_org == 1),
                    target_mask_2 = segment_ids_2_org,
                    attention_mask_2=input_mask_2_org,
                    token_type_ids=segment_ids_org,
                    attention_mask=input_mask_org,
                )
                logits_pos, pos_r = model(
                    input_ids_pos,
                    input_ids_2_pos,
                    target_mask=(segment_ids_pos == 1),
                    target_mask_2 = segment_ids_2_pos,
                    attention_mask_2 = input_mask_2_pos,
                    token_type_ids=segment_ids_pos,
                    attention_mask=input_mask_pos,
                )
                logits_neg, neg_r = model(
                    input_ids_neg,
                    input_ids_2_neg,
                    target_mask=(segment_ids_neg == 1),
                    target_mask_2 = segment_ids_2_neg,
                    attention_mask_2 = input_mask_2_neg,
                    token_type_ids=segment_ids_neg,
                    attention_mask=input_mask_neg,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits_org.view(-1, args.num_labels), label_ids_org.view(-1))
                pos_cos = cos(org_r, pos_r).reshape(org_r.shape[0], 1)
                neg_cos = cos(org_r, neg_r).reshape(org_r.shape[0], 1)
                v = torch.cat((pos_cos, neg_cos), dim = 1)
                exp = torch.exp(v)
                # we only need to take the log of the positive value over the sum of exp. 
                cts_loss = -torch.log(exp[:, 0]/torch.sum(exp, dim = 1)).sum()
            
            if args.r_drop:
                kl_loss = compute_kl_loss(logits_org, logits_pos)

            # average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()
                cts_loss = cts_loss.mean()
                if args.r_drop:
                    kl_loss = kl_loss.mean()
                
            loss +=  args.cts_loss_weight * cts_loss
            if args.r_drop:
                loss += args.r_drop_weight * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()
            step += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        # evaluate
        if args.do_eval:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval_cts(args, logger, model, eval_dataloader, all_guids, task_name)

            # update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    save_model(args, model, tokenizer)
                if args.task_name in ["vua", "magpie", "semeval", "vnc", "transfer", "combine"]:
                    save_model(args, model, tokenizer)

    logger.info(f"-----Best Result-----")
    for key in sorted(max_result.keys()):
        logger.info(f"  {key} = {str(max_result[key])}")

    return model, max_result

def evaluate_difficulty_cts(args, model, train_data):
    model.eval()
    eval_sampler = SequentialSampler(train_data)
    eval_dataloader = DataLoader(
        train_data,
        sampler=eval_sampler,
        batch_size=args.train_batch_size,
    )
    difficulties = []
    cos = torch.nn.CosineSimilarity(dim = 1)
    
    for eval_batch in tqdm(eval_dataloader, desc="Evaluating Difficulties", position=0, leave=True):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)
        
        if args.model_type in ["MELBERT_MIP", "MELBERT"]:
                input_ids_org, input_mask_org, segment_ids_org, input_ids_2_org, input_mask_2_org, segment_ids_2_org, label_ids_org, input_ids_pos, input_mask_pos, segment_ids_pos, input_ids_2_pos, input_mask_2_pos, segment_ids_2_pos, label_ids_pos, input_ids_neg, input_mask_neg, segment_ids_neg, input_ids_2_neg, input_mask_2_neg, segment_ids_2_neg, label_ids_neg = eval_batch
        else:
                input_ids_org, input_mask_org, segment_ids_org, label_ids_org, input_ids_pos, input_mask_pos, segment_ids_pos, label_ids_pos, input_ids_neg, input_mask_neg, segment_ids_neg, label_ids_neg = eval_batch


        with torch.no_grad():
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                logits_org, org_r = model(
                    input_ids_org,
                    target_mask=(segment_ids_org == 1),
                    token_type_ids=segment_ids_org,
                    attention_mask=input_mask_org,
                )
                logits_pos, pos_r = model(
                    input_ids_pos,
                    target_mask=(segment_ids_pos == 1),
                    token_type_ids=segment_ids_pos,
                    attention_mask=input_mask_pos,
                )
                logits_neg, neg_r = model(
                    input_ids_neg,
                    target_mask=(segment_ids_neg == 1),
                    token_type_ids=segment_ids_neg,
                    attention_mask=input_mask_neg,
                )
            
                pos_cos = cos(org_r, pos_r).reshape(org_r.shape[0], 1)
                neg_cos = cos(org_r, neg_r).reshape(org_r.shape[0], 1)
                v = torch.cat((pos_cos, neg_cos), dim = 1)
                exp = torch.exp(v)
                difficulty = (1-exp[:, 0])/(1-exp[:,1])
                difficulties += list(difficulty.detach().cpu().numpy())
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits_org, org_r = model(
                    input_ids_org,
                    input_ids_2_org,
                    target_mask=(segment_ids_org == 1),
                    target_mask_2 = segment_ids_2_org,
                    attention_mask_2=input_mask_2_org,
                    token_type_ids=segment_ids_org,
                    attention_mask=input_mask_org,
                )
                logits_pos, pos_r = model(
                    input_ids_pos,
                    input_ids_2_pos,
                    target_mask=(segment_ids_pos == 1),
                    target_mask_2 = segment_ids_2_pos,
                    attention_mask_2 = input_mask_2_pos,
                    token_type_ids=segment_ids_pos,
                    attention_mask=input_mask_pos,
                )
                logits_neg, neg_r = model(
                    input_ids_neg,
                    input_ids_2_neg,
                    target_mask=(segment_ids_neg == 1),
                    target_mask_2 = segment_ids_2_neg,
                    attention_mask_2 = input_mask_2_neg,
                    token_type_ids=segment_ids_neg,
                    attention_mask=input_mask_neg,
                )
                pos_cos = cos(org_r, pos_r).reshape(org_r.shape[0], 1)
                neg_cos = cos(org_r, neg_r).reshape(org_r.shape[0], 1)
                v = torch.cat((pos_cos, neg_cos), dim = 1)
                exp = torch.exp(v)
                difficulty = (1-exp[:, 0])/(1-exp[:,1])
                difficulties += list(difficulty.detach().cpu().numpy())
    return difficulties
    


def run_eval(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=False):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    pred_guids = []
    out_label_ids = None

    for eval_batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)

        if args.model_type in ["MELBERT_MIP", "MELBERT"]:
            (
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                idx,
                input_ids_2,
                input_mask_2,
                segment_ids_2,
            ) = eval_batch
        else:
            input_ids, input_mask, segment_ids, label_ids, idx = eval_batch

        with torch.no_grad():
            # compute loss values
            if args.model_type in ["BERT_BASE", "BERT_SEQ", "MELBERT_SPV"]:
                logits, _ = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits, _ = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    # compute metrics
    result = compute_metrics(preds, out_label_ids)

    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

    if return_preds:
        return preds
    return result

def run_eval_cts(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=False):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    pred_guids = []
    out_label_ids = None

    for eval_batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)

        if args.model_type in ["MELBERT_MIP", "MELBERT"]:
            input_ids, input_mask, segment_ids, label_ids, idx, input_ids_2, input_mask_2, segment_ids_2 = eval_batch
        else:
            input_ids, input_mask, segment_ids, label_ids, idx = eval_batch

        with torch.no_grad():
            # compute loss values
            if args.model_type in ["BERT_BASE", "BERT_SEQ", "MELBERT_SPV"]:
                logits, _ = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits, _ = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    # compute metrics
    result = compute_metrics(preds, out_label_ids)

    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

    if return_preds:
        return preds
    return result

def load_pretrained_model(args):
    # Pretrained Model
    bert = AutoModel.from_pretrained(args.bert_model)
    config = bert.config
    config.type_vocab_size = 4
    if "albert" in args.bert_model:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )
    else:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
    bert._init_weights(bert.embeddings.token_type_embeddings)
    bert.output_hidden_states = True

    # Additional Layers
    if args.model_type in ["BERT_BASE"] and args.cts:
        if args.lf:
            model = CTSModelForSequenceClassification_LF(
                args=args, Model=bert, config=config, num_labels=args.num_labels
            )
        else:
            model = CTSModelForSequenceClassification(
                args=args, Model=bert, config=config, num_labels=args.num_labels
            )
    if args.model_type == "BERT_SEQ" and args.cts:
        model = CTSModelForTokenClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_SPV" and args.cts:
        model = CTSModelForSequenceClassification_SPV(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_MIP" and args.cts:
        model = CTSModelForSequenceClassification_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT" and args.cts:
        if args.lf:
            model = CTSModelForSequenceClassification_SPV_MIP_LF(
                args=args, Model=bert, config=config, num_labels=args.num_labels
            )
        else:
            model = CTSModelForSequenceClassification_SPV_MIP(
                args=args, Model=bert, config=config, num_labels=args.num_labels
            )
            
    if args.model_type in ["BERT_BASE"] and not args.cts and args.cl:
        if args.lf:
            model = CTSModelForSequenceClassification_LF(
                args=args, Model=bert, config=config, num_labels=args.num_labels
            )
        else:
            model = CTSModelForSequenceClassification(
                args=args, Model=bert, config=config, num_labels=args.num_labels
            )
    if args.model_type == "BERT_SEQ" and not args.cts and args.cl:
        model = CTSModelForTokenClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_SPV" and not args.cts and args.cl:
        model = CTSModelForSequenceClassification_SPV(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_MIP" and not args.cts and args.cl:
        model = CTSModelForSequenceClassification_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT" and not args.cts and args.cl:
        if args.lf:
            model = CTSModelForSequenceClassification_SPV_MIP_LF(
                args=args, Model=bert, config=config, num_labels=args.num_labels
            )
        else:
            model = CTSModelForSequenceClassification_SPV_MIP(
                args=args, Model=bert, config=config, num_labels=args.num_labels
            )

    if args.model_type in ["BERT_BASE"] and not args.cts and not args.cl:
        model = AutoModelForSequenceClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "BERT_SEQ" and not args.cts and not args.cl:
        model = AutoModelForTokenClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_SPV" and not args.cts:
        model = AutoModelForSequenceClassification_SPV(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_MIP" and not args.cts and not args.cl:
        model = AutoModelForSequenceClassification_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT" and not args.cts and not args.cl:
        model = AutoModelForSequenceClassification_SPV_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )

    model.to(args.device)
    if args.n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)
    return model


def save_model(args, model, tokenizer):
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.log_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.log_dir)

    # Good practice: save your training arguments together with the trained model
    output_args_file = os.path.join(args.log_dir, ARGS_NAME)
    torch.save(args, output_args_file)


def load_trained_model(args, model, tokenizer):
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)

    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(output_model_file))
    else:
        model.load_state_dict(torch.load(output_model_file))

    return model


if __name__ == "__main__":
    main()