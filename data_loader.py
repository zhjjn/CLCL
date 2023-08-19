import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from run_classifier_dataset_utils import (
    convert_examples_to_two_features,
    convert_cts_examples_to_two_features,
    convert_examples_to_features,
    convert_two_examples_to_features,
    convert_cts_examples_to_features,
    convert_cts_examples_to_features_idm,
    convert_cts_examples_to_two_features_idm,
    convert_two_examples_to_features_idm,
    convert_examples_to_two_features_idm,
)


def load_train_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    # Prepare data loader
    if task_name == "vua":
        train_examples = processor.get_train_examples(args.data_dir)
    elif task_name == "trofi":
        train_examples = processor.get_train_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    # make features file
    if args.model_type == "BERT_BASE":
        train_features = convert_two_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        train_features = convert_examples_to_two_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    # make features into tensor
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # add additional features for MELBERT_MIP and MELBERT
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
        all_input_mask_2 = torch.tensor([f.input_mask_2 for f in train_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor(
            [f.segment_ids_2 for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    else:
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    return train_dataloader


def load_train_data_cts(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    # Prepare data loader
    if task_name == "vua":
        train_examples = processor.get_train_examples_cts(args.data_dir)
    elif task_name == "magpie":
        train_examples = processor.get_train_examples_cts(args.data_dir)
    elif task_name == "combine":
        train_examples = processor.get_train_examples_cts(args.data_dir)
    elif task_name == "transfer":
        train_examples = processor.get_train_examples_cts(args.data_dir)
    elif task_name == "semeval":
        train_examples = processor.get_train_examples_cts(args.data_dir)
    elif task_name == "vnc":
        train_examples = processor.get_train_examples_cts(args.data_dir)
    elif task_name == "trofi":
        train_examples = processor.get_train_examples_cts(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")
        
    
    if task_name in ['vua', 'trofi', 'mohx']:
        if args.model_type in ["BERT_BASE", "MELBERT_SPV"]:
            train_features_org = convert_cts_examples_to_features(
                train_examples, "Org", label_list, args.max_seq_length, tokenizer, output_mode
            )
            train_features_pos = convert_cts_examples_to_features(
                train_examples, "Pos", label_list, args.max_seq_length, tokenizer, output_mode
            )
            train_features_neg = convert_cts_examples_to_features(
                train_examples, "Neg", label_list, args.max_seq_length, tokenizer, output_mode
            )
        if args.model_type in ["BERT_SEQ"]:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
            )

        if args.model_type in ['MELBERT_MIP', "MELBERT"]:
            train_features_org = convert_cts_examples_to_two_features(
                train_examples, "Org", label_list, args.max_seq_length, tokenizer, output_mode, args
            )
            train_features_pos = convert_cts_examples_to_two_features(
                train_examples, "Pos", label_list, args.max_seq_length, tokenizer, output_mode, args
            )
            train_features_neg = convert_cts_examples_to_two_features(
                train_examples, "Neg", label_list, args.max_seq_length, tokenizer, output_mode, args
            )
    else:
        if args.model_type in ["BERT_BASE", "MELBERT_SPV"]:
            train_features_org = convert_cts_examples_to_features_idm(
                train_examples, "Org", label_list, args.max_seq_length, tokenizer, output_mode
            )
            train_features_pos = convert_cts_examples_to_features_idm(
                train_examples, "Pos", label_list, args.max_seq_length, tokenizer, output_mode
            )
            train_features_neg = convert_cts_examples_to_features_idm(
                train_examples, "Neg", label_list, args.max_seq_length, tokenizer, output_mode
            )
        if args.model_type in ['MELBERT_MIP', "MELBERT"]:
            train_features_org = convert_cts_examples_to_two_features_idm(
                train_examples, "Org", label_list, args.max_seq_length, tokenizer, output_mode, args
            )
            train_features_pos = convert_cts_examples_to_two_features_idm(
                train_examples, "Pos", label_list, args.max_seq_length, tokenizer, output_mode, args
            )
            train_features_neg = convert_cts_examples_to_two_features_idm(
                train_examples, "Neg", label_list, args.max_seq_length, tokenizer, output_mode, args
            )
    
    

    # make features into tensor
    all_input_ids_org = torch.tensor([f.input_ids for f in train_features_org], dtype=torch.long)
    all_input_mask_org = torch.tensor([f.input_mask for f in train_features_org], dtype=torch.long)
    all_segment_ids_org = torch.tensor([f.segment_ids for f in train_features_org], dtype=torch.long)
    all_label_ids_org = torch.tensor([f.label_id for f in train_features_org], dtype=torch.long)
    
    all_input_ids_pos = torch.tensor([f.input_ids for f in train_features_pos], dtype=torch.long)
    all_input_mask_pos = torch.tensor([f.input_mask for f in train_features_pos], dtype=torch.long)
    all_segment_ids_pos = torch.tensor([f.segment_ids for f in train_features_pos], dtype=torch.long)
    all_label_ids_pos = torch.tensor([f.label_id for f in train_features_org], dtype=torch.long)
    
    all_input_ids_neg = torch.tensor([f.input_ids for f in train_features_neg], dtype=torch.long)
    all_input_mask_neg= torch.tensor([f.input_mask for f in train_features_neg], dtype=torch.long)
    all_segment_ids_neg = torch.tensor([f.segment_ids for f in train_features_neg], dtype=torch.long)
    all_label_ids_neg = (torch.tensor([f.label_id for f in train_features_org], dtype=torch.long) - 1) ** 2

    # add additional features for MELBERT_MIP and MELBERT
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids_2_org = torch.tensor([f.input_ids_2 for f in train_features_org], dtype=torch.long)
        all_input_mask_2_org = torch.tensor([f.input_mask_2 for f in train_features_org], dtype=torch.long)
        all_segment_ids_2_org = torch.tensor(
            [f.segment_ids_2 for f in train_features_org], dtype=torch.long
        )
        all_input_ids_2_pos = torch.tensor([f.input_ids_2 for f in train_features_pos], dtype=torch.long)
        all_input_mask_2_pos = torch.tensor([f.input_mask_2 for f in train_features_pos], dtype=torch.long)
        all_segment_ids_2_pos = torch.tensor(
            [f.segment_ids_2 for f in train_features_pos], dtype=torch.long
        )
        all_input_ids_2_neg = torch.tensor([f.input_ids_2 for f in train_features_neg], dtype=torch.long)
        all_input_mask_2_neg = torch.tensor([f.input_mask_2 for f in train_features_neg], dtype=torch.long)
        all_segment_ids_2_neg = torch.tensor(
            [f.segment_ids_2 for f in train_features_neg], dtype=torch.long
        )
        train_data = TensorDataset(
            all_input_ids_org,
            all_input_mask_org,
            all_segment_ids_org,
            all_input_ids_2_org,
            all_input_mask_2_org,
            all_segment_ids_2_org,
            all_label_ids_org,
            all_input_ids_pos,
            all_input_mask_pos,
            all_segment_ids_pos,
            all_input_ids_2_pos,
            all_input_mask_2_pos,
            all_segment_ids_2_pos,
            all_label_ids_pos,
            all_input_ids_neg,
            all_input_mask_neg,
            all_segment_ids_neg,
            all_input_ids_2_neg,
            all_input_mask_2_neg,
            all_segment_ids_2_neg,
            all_label_ids_neg,
        )
    else:
        train_data = TensorDataset(all_input_ids_org,
                                   all_input_mask_org, 
                                   all_segment_ids_org, 
                                   all_label_ids_org,
                                   all_input_ids_pos,
                                   all_input_mask_pos, 
                                   all_segment_ids_pos, 
                                   all_label_ids_pos,
                                   all_input_ids_neg,
                                   all_input_mask_neg, 
                                   all_segment_ids_neg, 
                                   all_label_ids_neg,
                                  )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    return train_data, train_dataloader, train_examples


def load_train_data_kf(
    args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None
):
    # Prepare data loader
    if task_name == "vua":
        train_examples = processor.get_train_examples(args.data_dir)
    elif task_name == "trofi":
        train_examples = processor.get_train_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    # make features file
    if args.model_type == "BERT_BASE":
        train_features = convert_two_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        train_features = convert_examples_to_two_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    # make features into tensor
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # add additional features for MELBERT_MIP and MELBERT
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
        all_input_mask_2 = torch.tensor([f.input_mask_2 for f in train_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor(
            [f.segment_ids_2 for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    else:
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    gkf = StratifiedKFold(n_splits=args.num_bagging).split(X=all_input_ids, y=all_label_ids.numpy())
    return train_data, gkf


def load_test_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    if task_name == "vua":
        eval_examples = processor.get_test_examples(args.data_dir)
    elif task_name == "trofi":
        eval_examples = processor.get_test_examples(args.data_dir,k)
    elif task_name == "magpie":
        eval_examples = processor.get_test_examples(args.data_dir)
    elif task_name == "combine":
        eval_examples = processor.get_test_examples(args.data_dir)
    elif task_name == "transfer":
        eval_examples = processor.get_test_examples(args.data_dir)
    elif task_name == "semeval":
        eval_examples = processor.get_test_examples(args.data_dir)
    elif task_name == "vnc":
        eval_examples = processor.get_test_examples(args.data_dir)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")
        
    
    if task_name in ['vua', 'trofi', 'mohx']:
        if args.model_type == "BERT_BASE":
            eval_features = convert_two_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode
            )
        if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
            )
        if args.model_type in ["MELBERT_MIP", "MELBERT"]:
            eval_features = convert_examples_to_two_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
            )
    else:
        if args.model_type == "BERT_BASE":
            eval_features = convert_two_examples_to_features_idm(
                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode
            )
        if args.model_type in ["MELBERT_MIP", "MELBERT"]:
            eval_features = convert_examples_to_two_features_idm(
                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
            )
    

    logger.info("***** Running evaluation *****")
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_guids = [f.guid for f in eval_features]
        all_idx = torch.tensor([i for i in range(len(eval_features))], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in eval_features], dtype=torch.long)
        all_input_mask_2 = torch.tensor([f.input_mask_2 for f in eval_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_idx,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    else:
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_guids = [f.guid for f in eval_features]
        all_idx = torch.tensor([i for i in range(len(eval_features))], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_idx
        )

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return all_guids, eval_dataloader