import re
import os
import random
import tarfile
import urllib
from torchtext import data
from torchtext.data import Field


def ir_dataset(TEXT, args):
    print(TEXT.vocab.stoi)
    tv_datafields = [("query", TEXT), ("doc", TEXT)]
    trn, vld = data.TabularDataset.splits(
                   path=args.data_path, # the root directory where the data lies
                   train='train.tsv', validation="dev.tsv", 
                   format='tsv',
                   skip_header=True, 
                   fields=tv_datafields)

    # iterator
    train_iter, val_iter = data.BucketIterator.splits(
         (trn, vld), 
         batch_sizes= (args.batch_size, args.batch_size),
         sort_key=lambda x: len(x.doc), 
         sort_within_batch=False,
         sort=False, 
         shuffle = True,
         repeat=False
    )

    return train_iter, val_iter

def stance_dataset(TEXT, LABEL, args, testing = True):
    tv_datafields = [("label", LABEL), ("query", TEXT), ("doc", TEXT)]
    trn, vld = data.TabularDataset.splits(
                   path=args.stance_data_path, # the root directory where the stance data lies
                   train='train.tsv', validation="dev.tsv", 
                   format='tsv',
                   skip_header=True,
                   fields=tv_datafields)

    TEXT.build_vocab(trn)

    # iterator
    train_iter = data.BucketIterator(
         trn, 
         batch_size= args.batch_size,
         sort_key=lambda x: len(x.doc), 
         sort_within_batch=False,
         sort=False, 
         shuffle = True,
         repeat=False
    )

    val_iter = data.Iterator(
          vld,
          batch_size=args.batch_size*4,
          shuffle = False, 
          sort_within_batch=False, 
          repeat=False)

    if testing:
      test_datafields = [("query", TEXT), ("doc", TEXT)]
      test = data.TabularDataset(
                   path=os.path.join(args.stance_data_path, "test.tsv"),
                   format='tsv',
                   skip_header=True,
                   fields=test_datafields)
      test_iter = data.Iterator(
            test,
            batch_size=args.batch_size*4,
            shuffle = False, 
            sort_within_batch=False, 
            repeat=False)
      return train_iter, val_iter, test_iter
    
    else:
      return train_iter, val_iter
