# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import sys
import time

import numpy as np
from loguru import logger

sys.path.append('..')
from text2vec import CosentModel, SentenceBertModel, BertMatchModel
from text2vec import CosentTrainDataset, TextMatchingTestDataset, TextMatchingTrainDataset
from text2vec import cos_sim, compute_spearmanr, EncoderType


def calc_similarity_scores(model, sents1, sents2, labels):
    t1 = time.time()
    e1 = model.encode(sents1)
    e2 = model.encode(sents2)
    spend_time = time.time() - t1
    s = cos_sim(e1, e2)
    sims = []
    for i in range(len(sents1)):
        sims.append(s[i][i])
    sims = np.array(sims)
    labels = np.array(labels)
    spearman = compute_spearmanr(labels, sims)
    logger.debug(f'labels: {labels[:10]}')
    logger.debug(f'preds:  {sims[:10]}')
    logger.debug(f'Spearman: {spearman}')
    logger.debug(
        f'spend time: {spend_time:.4f}, count:{len(sents1 + sents2)}, qps: {len(sents1 + sents2) / spend_time}')
    return spearman


def load_en_stsb_dataset(stsb_file):
    # Convert the dataset to a DataLoader ready for training
    logger.info("Read STSbenchmark dataset")
    train_samples = []
    valid_samples = []
    test_samples = []
    with gzip.open(stsb_file, 'rt', encoding='utf8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score'])
            if row['split'] == 'dev':
                valid_samples.append((row['sentence1'], row['sentence2'], score))
            elif row['split'] == 'test':
                test_samples.append((row['sentence1'], row['sentence2'], score))
            else:
                score = int(score > 2.5)
                train_samples.append((row['sentence1'], row['sentence2'], score))
    return train_samples, valid_samples, test_samples


def convert_to_cosent_train_dataset(train_samples):
    # Convert the dataset to CoSENT model training format
    train_dataset = []
    for sample in train_samples:
        if len(sample) != 3:
            continue
        train_dataset.append((sample[0], sample[2]))
        train_dataset.append((sample[1], sample[2]))
    return train_dataset


def main():
    parser = argparse.ArgumentParser('Text Matching task')
    parser.add_argument('--model_arch', default='cosent', const='cosent', nargs='?',
                        choices=['cosent', 'sentencebert', 'bert'], help='model architecture')
    parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='name of transformers model')
    parser.add_argument('--stsb_file', default='data/English-STS-B/stsbenchmark.tsv.gz', type=str,
                        help='Train data path')
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict.")
    parser.add_argument('--output_dir', default='./outputs/STS-B-en-model', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--encoder_type', default='FIRST_LAST_AVG', type=lambda t: EncoderType[t],
                        choices=list(EncoderType), help='Encoder type, string name of EncoderType')
    args = parser.parse_args()
    logger.info(args)
    train_samples, valid_samples, test_samples = load_en_stsb_dataset(args.stsb_file)

    if args.do_train:
        if args.model_arch == 'cosent':
            model = CosentModel(model_name_or_path=args.model_name, encoder_type=args.encoder_type,
                                max_seq_length=args.max_seq_length)
            train_samples = convert_to_cosent_train_dataset(train_samples)
            train_dataset = CosentTrainDataset(model.tokenizer, train_samples, args.max_seq_length)
        elif args.model_arch == 'sentencebert':
            model = SentenceBertModel(model_name_or_path=args.model_name, encoder_type=args.encoder_type,
                                      max_seq_length=args.max_seq_length)
            train_dataset = TextMatchingTrainDataset(model.tokenizer, train_samples, args.max_seq_length)
        else:
            model = BertMatchModel(model_name_or_path=args.model_name, encoder_type=args.encoder_type,
                                   max_seq_length=args.max_seq_length)
            train_dataset = TextMatchingTrainDataset(model.tokenizer, train_samples, args.max_seq_length)
        valid_dataset = TextMatchingTestDataset(model.tokenizer, valid_samples, args.max_seq_length)
        model.train(train_dataset,
                    args.output_dir,
                    eval_dataset=valid_dataset,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    lr=args.learning_rate)
        logger.info(f"Model saved to {args.output_dir}")

    if args.do_predict:
        if args.model_arch == 'cosent':
            model = CosentModel(model_name_or_path=args.output_dir, encoder_type=args.encoder_type,
                                max_seq_length=args.max_seq_length)
        elif args.model_arch == 'sentencebert':
            model = SentenceBertModel(model_name_or_path=args.output_dir, encoder_type=args.encoder_type,
                                      max_seq_length=args.max_seq_length)
        else:
            model = BertMatchModel(model_name_or_path=args.output_dir, encoder_type=args.encoder_type,
                                   max_seq_length=args.max_seq_length)
        # Predict embeddings
        srcs = []
        trgs = []
        labels = []
        for terms in test_samples:
            src, trg, label = terms[0], terms[1], terms[2]
            srcs.append(src)
            trgs.append(trg)
            labels.append(label)
        logger.debug(f'{test_samples[0]}')
        sentence_embeddings = model.encode(srcs)
        logger.debug(f"{type(sentence_embeddings)}, {sentence_embeddings.shape}, {sentence_embeddings[0].shape}")
        # Predict similarity scores
        calc_similarity_scores(model, srcs, trgs, labels)


if __name__ == '__main__':
    main()
