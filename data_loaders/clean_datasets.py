from pathlib import Path
from typing import Dict
from collections import defaultdict
import random
import pickle
import numpy as np
import re

from .load import _get_uniques_, _pad_statements_, count_stats, remove_dups

def load_clean_wikipeople_statements(subtype, maxlen=17) -> Dict:
    """
        :return: train/valid/test splits for the wikipeople dataset in its quints form
    """
    DIRNAME = Path('./data/clean/wikipeople')

    # Load raw shit
    with open(DIRNAME / 'train.txt', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(line.strip("\n").split(","))

    with open(DIRNAME / 'test.txt', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(line.strip("\n").split(","))

    with open(DIRNAME / 'valid.txt', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(line.strip("\n").split(","))

    # Get uniques
    statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                             test_data=raw_tst,
                                                             valid_data=raw_val)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in raw_trn:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in raw_val:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in raw_tst:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    if subtype == "triples":
        maxlen = 3
    elif subtype == "quints":
        maxlen = 5

    train, valid, test = _pad_statements_(train, maxlen), \
                         _pad_statements_(valid, maxlen), \
                         _pad_statements_(test, maxlen)

    if subtype == "triples" or subtype == "quints":
        train, valid, test = remove_dups(train), remove_dups(valid), remove_dups(test)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}

def load_clean_jf17k_statements(subtype, maxlen=15) -> Dict:
    PARSED_DIR = Path('./data/clean/jf17k')

    training_statements = []
    test_statements = []

    with open(PARSED_DIR / 'train.txt', 'r') as train_file, \
        open(PARSED_DIR / 'test.txt', 'r') as test_file:

        for line in train_file:
            training_statements.append(line.strip("\n").split(","))

        for line in test_file:
            test_statements.append(line.strip("\n").split(","))

    st_entities, st_predicates = _get_uniques_(training_statements, test_statements, test_statements)
    st_entities = ['__na__'] + st_entities
    st_predicates = ['__na__'] + st_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    # sample valid as 20% of train
    random.shuffle(training_statements)
    tr_st = training_statements[:int(0.8*len(training_statements))]
    val_st = training_statements[int(0.8*len(training_statements)):]

    train, valid, test = [], [], []
    for st in tr_st:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)

    for st in val_st:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)

    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)


    if subtype == "triples":
        maxlen = 3
    elif subtype == "quints":
        maxlen = 5

    train, valid, test = _pad_statements_(train, maxlen), \
                         _pad_statements_(valid, maxlen), \
                         _pad_statements_(test, maxlen)

    if subtype == "triples" or subtype == "quints":
        train, valid, test = remove_dups(train), remove_dups(valid), remove_dups(test)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_clean_wd50k(name, subtype, maxlen=43) -> Dict:
    """
        :return: train/valid/test splits for the wd50k datasets
    """
    assert name in ['wd50k', 'wd50k_100', 'wd50k_33', 'wd50k_66'], \
        "Incorrect dataset"
    assert subtype in ["triples", "quints", "statements"], "Incorrect subtype: triples/quints/statements"

    DIRNAME = Path(f'./data/clean/{name}/{subtype}')

    # Load raw shit
    with open(DIRNAME / 'train.txt', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(line.strip("\n").split(","))

    with open(DIRNAME / 'test.txt', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(line.strip("\n").split(","))

    with open(DIRNAME / 'valid.txt', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(line.strip("\n").split(","))

    # Get uniques
    statement_entities, statement_predicates = _get_uniques_(train_data=raw_trn,
                                                             test_data=raw_tst,
                                                             valid_data=raw_val)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in raw_trn:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in raw_val:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in raw_tst:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    if subtype != "triples":
        if subtype == "quints":
            maxlen = 5
        train, valid, test = _pad_statements_(train, maxlen), \
                             _pad_statements_(valid, maxlen), \
                             _pad_statements_(test, maxlen)

    if subtype == "triples" or subtype == "quints":
        train, valid, test = remove_dups(train), remove_dups(valid), remove_dups(test)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


if __name__ == "__main__":
    count_stats(load_clean_wd50k("wd50k","statements",43))