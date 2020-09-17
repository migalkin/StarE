"""
    File which enables easily loading any dataset we need
"""
import json
from tqdm import tqdm
from functools import partial
from typing import List, Union, Dict, Callable
import numpy as np
import pickle
from utils.utils import PARSED_DATA_DIR, KNOWN_DATASETS
from pathlib import Path
from utils.utils_mytorch import FancyDict
import random
#from utils import *

# KNOWN_DATASETS = ['fb15k237', 'wd50k', 'fb15k', 'wikipeople', 'wd50k_100']
# RAW_DATA_DIR = Path('./data/raw_data')
# PARSED_DATA_DIR = Path('./data/parsed_data')
# PRETRAINING_DATA_DIR = Path('./data/pre_training_data')

def _conv_to_our_format_(data, filter_literals=True):
    conv_data = []
    dropped_statements = 0
    dropped_quals = 0
    for datum in tqdm(data):
        try:
            conv_datum = []

            # Get head and tail rels
            head, tail, rel_h, rel_t = None, None, None, None
            for rel, val in datum.items():
                if rel[-2:] == '_h' and type(val) is str:
                    head = val
                    rel_h = rel[:-2]
                if rel[-2:] == '_t' and type(val) is str:
                    tail = val
                    rel_t = rel[:-2]
                    if filter_literals and "http://" in tail:
                        dropped_statements += 1
                        raise Exception

            assert head and tail and rel_h and rel_t, f"Weird data point. Some essentials not found. Quitting\nD:{datum}"
            assert rel_h == rel_t, f"Weird data point. Head and Tail rels are different. Quitting\nD: {datum}"

            # Drop this bs
            datum.pop(rel_h + '_h')
            datum.pop(rel_t + '_t')
            datum.pop('N')
            conv_datum += [head, rel_h, tail]

            # Get all qualifiers
            for k, v in datum.items():
                for _v in v:
                    if filter_literals and "http://" in _v:
                        dropped_quals += 1
                        continue
                    conv_datum += [k, _v]

            conv_data.append(tuple(conv_datum))
        except Exception:
            continue
    print(f"\n Dropped {dropped_statements} statements and {dropped_quals} quals with literals \n ")
    return conv_data


def _conv_to_our_quint_format_(data, filter_literals=True):
    conv_data = []
    dropped_statements = 0
    dropped_quals = 0
    for datum in tqdm(data):
        try:
            conv_datum = []

            # Get head and tail rels
            head, tail, rel_h, rel_t = None, None, None, None
            for rel, val in datum.items():
                if rel[-2:] == '_h' and type(val) is str:
                    head = val
                    rel_h = rel[:-2]
                if rel[-2:] == '_t' and type(val) is str:
                    tail = val
                    rel_t = rel[:-2]
                    if filter_literals and "http://" in tail:
                        dropped_statements += 1
                        raise Exception

            assert head and tail and rel_h and rel_t, f"Weird data point. Some essentials not found. Quitting\nD:{datum}"
            assert rel_h == rel_t, f"Weird data point. Head and Tail rels are different. Quitting\nD: {datum}"

            # Drop this bs
            datum.pop(rel_h + '_h')
            datum.pop(rel_t + '_t')
            datum.pop('N')
            conv_datum += [head, rel_h, tail, None, None]

            if len(datum.items()) == 0:
                conv_data.append(tuple(conv_datum))
            else:
                # Get all qualifiers
                for k, v in datum.items():
                    conv_datum[3] = k
                    for _v in v:
                        if filter_literals and "http://" in _v:
                            dropped_quals += 1
                            continue
                        conv_datum[4] = _v
                        conv_data.append(tuple(conv_datum))

        except Exception:
            continue
    print(f"\n Dropped {dropped_statements} statements and {dropped_quals} quals with literals \n ")
    return conv_data


def _conv_jf17k_to_quints(data):
    result = []
    for statement in data:
        ents = statement[0::2]
        rels = statement[1::2]

        if len(rels) == 1:
            result.append(statement)
        else:
            s, p, o = statement[0], statement[1], statement[2]
            qual_rel = rels[1:]
            qual_ent = ents[2:]
            for i in range(len(qual_rel)):
                result.append([s, p, o, qual_rel[i], qual_ent[i]])

    return result


def _get_uniques_(train_data: List[tuple], valid_data: List[tuple], test_data: List[tuple]) -> (
list, list):
    """ Throw in parsed_data/wd50k/ files and we'll count the entities and predicates"""

    statement_entities, statement_predicates = [], []

    for statement in train_data + valid_data + test_data:
        statement_entities += statement[::2]
        statement_predicates += statement[1::2]

    statement_entities = sorted(list(set(statement_entities)))
    statement_predicates = sorted(list(set(statement_predicates)))

    return statement_entities, statement_predicates


def _pad_statements_(data: List[list], maxlen: int) -> List[list]:
    """ Padding index is always 0 as in the embedding layers of models. Cool? Cool. """
    result = [
        statement + [0] * (maxlen - len(statement)) if len(statement) < maxlen else statement[
                                                                                    :maxlen] for
        statement in data]
    return result

def clean_literals(data: List[list]) -> List[list]:
    """

    :param data: triples [s, p, o] with possible literals
    :return: triples [s,p,o] without literals

    """
    result = []
    for triple in data:
        if "http://" not in triple[2]:
            result.append(triple)

    return result

def remove_dups(data: List[list]) -> List[list]:
    """

    :param data: list of lists with possible duplicates
    :return: a list without duplicates
    """
    new_l = []
    for datum in tqdm(data):
        if datum not in new_l:
            new_l.append(datum)

    return new_l

def load_wd50k_quints() -> Dict:
    """

    :return:
    """

    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k'
    with open(wd50k_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(wd50k_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(wd50k_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = sorted(list(set(quints_entities)))
    quints_predicates = sorted(list(set(quints_predicates)))

    q_entities = ['__na__'] + quints_entities
    q_predicates = ['__na__'] + quints_predicates

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(q_entities)}
    prtoid = {pred: i for i, pred in enumerate(q_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(q_entities),
            "n_relations": len(q_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_triples() -> Dict:
    """

    :return:
    """

    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k'

    with open(wd50k_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(wd50k_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(wd50k_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_statements(maxlen: int) -> Dict:
    """
        Pull up data from parsed data (thanks magic mike!) and preprocess it to death.
    :return: dict
    """

    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k'
    with open(wd50k_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(wd50k_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(wd50k_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_33_quints() -> Dict:
    """

    :return:
    """

    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_33'
    with open(wd50k_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(wd50k_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(wd50k_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = sorted(list(set(quints_entities)))
    quints_predicates = sorted(list(set(quints_predicates)))

    q_entities = ['__na__'] + quints_entities
    q_predicates = ['__na__'] + quints_predicates

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(q_entities)}
    prtoid = {pred: i for i, pred in enumerate(q_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(q_entities),
            "n_relations": len(q_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_33_triples() -> Dict:
    """

    :return:
    """

    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_33'

    with open(wd50k_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(wd50k_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(wd50k_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_33_statements(maxlen: int) -> Dict:
    """
        Pull up data from parsed data (thanks magic mike!) and preprocess it to death.
    :return: dict
    """

    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_33'
    with open(wd50k_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(wd50k_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(wd50k_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_66_quints() -> Dict:
    """

    :return:
    """

    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_66'
    with open(wd50k_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(wd50k_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(wd50k_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = sorted(list(set(quints_entities)))
    quints_predicates = sorted(list(set(quints_predicates)))

    q_entities = ['__na__'] + quints_entities
    q_predicates = ['__na__'] + quints_predicates

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(q_entities)}
    prtoid = {pred: i for i, pred in enumerate(q_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(q_entities),
            "n_relations": len(q_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_66_triples() -> Dict:
    """

    :return:
    """

    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_66'

    with open(wd50k_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(wd50k_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(wd50k_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_66_statements(maxlen: int) -> Dict:
    """
        Pull up data from parsed data (thanks magic mike!) and preprocess it to death.
    :return: dict
    """

    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_66'
    with open(wd50k_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(wd50k_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(wd50k_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_100_statements(maxlen: int) -> Dict:
    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_100'
    with open(wd50k_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(wd50k_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(wd50k_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_100_quints() -> Dict:
    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_100'
    with open(wd50k_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(wd50k_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(wd50k_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = ['__na__'] + sorted(list(set(quints_entities)))
    quints_predicates = ['__na__'] + sorted(list(set(quints_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(quints_entities)}
    prtoid = {pred: i for i, pred in enumerate(quints_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]],
              entoid[q[4]]] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]],
              entoid[q[4]]] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]],
             entoid[q[4]]] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(quints_entities),
            "n_relations": len(quints_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_100_triples() -> Dict:
    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_100'

    with open(wd50k_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(wd50k_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(wd50k_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_100_33_statements(maxlen: int) -> Dict:
    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_100_33'
    with open(wd50k_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(wd50k_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(wd50k_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_100_33_quints() -> Dict:
    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_100_33'
    with open(wd50k_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(wd50k_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(wd50k_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = ['__na__'] + sorted(list(set(quints_entities)))
    quints_predicates = ['__na__'] + sorted(list(set(quints_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(quints_entities)}
    prtoid = {pred: i for i, pred in enumerate(quints_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(quints_entities),
            "n_relations": len(quints_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_100_33_triples() -> Dict:
    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_100_33'

    with open(wd50k_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(wd50k_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(wd50k_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_100_66_statements(maxlen: int) -> Dict:
    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_100_66'
    with open(wd50k_DIR / 'train_statements.pkl', 'rb') as f:
        train_statements = pickle.load(f)
    with open(wd50k_DIR / 'valid_statements.pkl', 'rb') as f:
        valid_statements = pickle.load(f)
    with open(wd50k_DIR / 'test_statements.pkl', 'rb') as f:
        test_statements = pickle.load(f)

    statement_entities, statement_predicates = _get_uniques_(train_data=train_statements,
                                                             valid_data=valid_statements,
                                                             test_data=test_statements)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in train_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in valid_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in test_statements:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_100_66_quints() -> Dict:
    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_100_66'
    with open(wd50k_DIR / 'train_quints.pkl', 'rb') as f:
        train_quints = pickle.load(f)
    with open(wd50k_DIR / 'valid_quints.pkl', 'rb') as f:
        valid_quints = pickle.load(f)
    with open(wd50k_DIR / 'test_quints.pkl', 'rb') as f:
        test_quints = pickle.load(f)

    quints_entities, quints_predicates = [], []

    for quint in train_quints + valid_quints + test_quints:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = ['__na__'] + sorted(list(set(quints_entities)))
    quints_predicates = ['__na__'] + sorted(list(set(quints_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(quints_entities)}
    prtoid = {pred: i for i, pred in enumerate(quints_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in train_quints]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in valid_quints]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in test_quints]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(quints_entities),
            "n_relations": len(quints_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wd50k_100_66_triples() -> Dict:
    # Load data from disk
    wd50k_DIR = PARSED_DATA_DIR / 'wd50k_100_66'

    with open(wd50k_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(wd50k_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(wd50k_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wikipeople_quints(filter_literals=True):
    # Load data from disk
    DIRNAME = Path('./data/raw_data/wikipeople')

    # Load raw shit
    with open(DIRNAME / 'n-ary_train.json', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(json.loads(line))

    with open(DIRNAME / 'n-ary_test.json', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(json.loads(line))

    with open(DIRNAME / 'n-ary_valid.json', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(json.loads(line))

    # raw_trn[:-10], raw_tst[:10], raw_val[:10]
    # Conv data to our format
    conv_trn, conv_tst, conv_val = _conv_to_our_quint_format_(raw_trn, filter_literals=filter_literals), \
                                   _conv_to_our_quint_format_(raw_tst, filter_literals=filter_literals), \
                                   _conv_to_our_quint_format_(raw_val, filter_literals=filter_literals)

    # quints_entities, quints_predicates = _get_uniques_(train_data=conv_trn,
    #                                                          test_data=conv_tst,
    #                                                          valid_data=conv_val)

    # st_entities = ['__na__'] + quints_entities
    # st_predicates = ['__na__'] + quints_predicates
    # quints_entities = ['__na__'] + sorted(list(set(quints_entities)))
    # quints_predicates = ['__na__'] + sorted(list(set(quints_predicates)))
    quints_entities, quints_predicates = [], []
    for quint in conv_trn + conv_val + conv_tst:
        quints_entities += [quint[0], quint[2]]
        if quint[4]:
            quints_entities.append(quint[4])

        quints_predicates.append(quint[1])
        if quint[3]:
            quints_predicates.append(quint[3])

    quints_entities = sorted(list(set(quints_entities)))
    quints_predicates = sorted(list(set(quints_predicates)))

    q_entities = ['__na__'] + quints_entities
    q_predicates = ['__na__'] + quints_predicates

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(q_entities)}
    prtoid = {pred: i for i, pred in enumerate(q_predicates)}

    train = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in conv_trn]
    valid = [[entoid[q[0]],
              prtoid[q[1]],
              entoid[q[2]],
              prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
              entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in conv_val]
    test = [[entoid[q[0]],
             prtoid[q[1]],
             entoid[q[2]],
             prtoid[q[3]] if q[3] is not None else prtoid['__na__'],
             entoid[q[4]] if q[4] is not None else entoid['__na__']] for q in conv_tst]

    return {"train": train, "valid": valid, "test": test, "n_entities": len(q_entities),
            "n_relations": len(q_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wikipeople_triples(filter_literals=True):
    # Load data from disk
    WP_DIR = PARSED_DATA_DIR / 'wikipeople'

    with open(WP_DIR / 'train_triples.pkl', 'rb') as f:
        train_triples = pickle.load(f)
    with open(WP_DIR / 'valid_triples.pkl', 'rb') as f:
        valid_triples = pickle.load(f)
    with open(WP_DIR / 'test_triples.pkl', 'rb') as f:
        test_triples = pickle.load(f)

    triples_entities, triples_predicates = [], []

    if filter_literals:
        train_triples = clean_literals(train_triples)
        valid_triples = clean_literals(valid_triples)
        test_triples = clean_literals(test_triples)


    for triple in train_triples + valid_triples + test_triples:
        triples_entities += [triple[0], triple[2]]
        triples_predicates.append(triple[1])

    triples_entities = ['__na__'] + sorted(list(set(triples_entities)))
    triples_predicates = ['__na__'] + sorted(list(set(triples_predicates)))

    # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
    entoid = {pred: i for i, pred in enumerate(triples_entities)}
    prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

    train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in train_triples]
    valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in valid_triples]
    test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_triples]


    return {"train": train, "valid": valid, "test": test, "n_entities": len(triples_entities),
            "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_wikipeople_statements(maxlen=17, filter_literals=True) -> Dict:
    """
        :return: train/valid/test splits for the wikipeople dataset in its quints form
    """
    DIRNAME = Path('./data/raw_data/wikipeople')

    with open(DIRNAME / 'n-ary_train.json', 'r') as f:
        raw_trn = []
        for line in f.readlines():
            raw_trn.append(json.loads(line))

    with open(DIRNAME / 'n-ary_test.json', 'r') as f:
        raw_tst = []
        for line in f.readlines():
            raw_tst.append(json.loads(line))

    with open(DIRNAME / 'n-ary_valid.json', 'r') as f:
        raw_val = []
        for line in f.readlines():
            raw_val.append(json.loads(line))

    # raw_trn[:-10], raw_tst[:10], raw_val[:10]
    # Conv data to our format
    conv_trn, conv_tst, conv_val = _conv_to_our_format_(raw_trn, filter_literals=filter_literals), \
                                   _conv_to_our_format_(raw_tst, filter_literals=filter_literals), \
                                   _conv_to_our_format_(raw_val, filter_literals=filter_literals)
    #conv_trn, conv_tst, conv_val = remove_dups(conv_trn), remove_dups(conv_tst), remove_dups(conv_val)
    # Get uniques
    statement_entities, statement_predicates = _get_uniques_(train_data=conv_trn,
                                                             test_data=conv_tst,
                                                             valid_data=conv_val)

    st_entities = ['__na__'] + statement_entities
    st_predicates = ['__na__'] + statement_predicates

    entoid = {pred: i for i, pred in enumerate(st_entities)}
    prtoid = {pred: i for i, pred in enumerate(st_predicates)}

    train, valid, test = [], [], []
    for st in conv_trn:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        train.append(id_st)
    for st in conv_val:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        valid.append(id_st)
    for st in conv_tst:
        id_st = []
        for i, uri in enumerate(st):
            id_st.append(entoid[uri] if i % 2 is 0 else prtoid[uri])
        test.append(id_st)

    train, valid, test = _pad_statements_(train, maxlen), _pad_statements_(valid,
                                                                           maxlen), _pad_statements_(
        test,
        maxlen)

    return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
            "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_jf17k_triples() -> Dict:
    PARSED_DIR = Path('./data/parsed_data/jf17k')

    training_statements = []
    test_statements = []

    with open(PARSED_DIR / 'train.txt', 'r') as train_file, \
            open(PARSED_DIR / 'test.txt', 'r') as test_file:

        for line in train_file:
            training_statements.append(line.strip("\n").split(","))

        for line in test_file:
            test_statements.append(line.strip("\n").split(","))

        entities, predicates = [], []
        for s in training_statements + test_statements:
            entities += [s[0], s[2]]
            predicates.append(s[1])

        triples_entities = ['__na__'] + sorted(list(set(entities)))
        triples_predicates = ['__na__'] + sorted(list(set(predicates)))

        # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
        entoid = {pred: i for i, pred in enumerate(triples_entities)}
        prtoid = {pred: i for i, pred in enumerate(triples_predicates)}

        # sample valid as 20% of train
        random.shuffle(training_statements)
        tr_st = training_statements[:int(0.8 * len(training_statements))]
        val_st = training_statements[int(0.8 * len(training_statements)):]

        train = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in tr_st]
        valid = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in val_st]
        test = [[entoid[q[0]], prtoid[q[1]], entoid[q[2]]] for q in test_statements]

        # optional
        clean_train, clean_valid, clean_test = remove_dups(train), remove_dups(valid), remove_dups(test)

        return {"train": clean_train, "valid": clean_valid, "test": clean_test, "n_entities": len(triples_entities),
                "n_relations": len(triples_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_jf17k_quints() -> Dict:
    PARSED_DIR = Path('./data/parsed_data/jf17k')

    training_statements = []
    test_statements = []

    with open(PARSED_DIR / 'train.txt', 'r') as train_file, \
            open(PARSED_DIR / 'test.txt', 'r') as test_file:

        for line in train_file:
            training_statements.append(line.strip("\n").split(","))

        for line in test_file:
            test_statements.append(line.strip("\n").split(","))

        # sample valid as 20% of train
        random.shuffle(training_statements)
        tr_st = training_statements[:int(0.8 * len(training_statements))]
        val_st = training_statements[int(0.8 * len(training_statements)):]

        train_quints = _conv_jf17k_to_quints(tr_st)
        val_quints = _conv_jf17k_to_quints(val_st)
        test_quints = _conv_jf17k_to_quints(test_statements)

        quints_entities, quints_predicates = [], []
        for quint in train_quints + val_quints + test_quints:
            quints_entities += [quint[0], quint[2]]
            quints_predicates.append(quint[1])
            if len(quint) > 3:
                quints_entities.append(quint[4])
                quints_predicates.append(quint[3])

        quints_entities = sorted(list(set(quints_entities)))
        quints_predicates = sorted(list(set(quints_predicates)))

        q_entities = ['__na__'] + quints_entities
        q_predicates = ['__na__'] + quints_predicates

        # uritoid = {ent: i for i, ent in enumerate(['__na__', '__pad__'] + entities +  predicates)}
        entoid = {pred: i for i, pred in enumerate(q_entities)}
        prtoid = {pred: i for i, pred in enumerate(q_predicates)}

        train = [[entoid[q[0]],
                  prtoid[q[1]],
                  entoid[q[2]],
                  prtoid[q[3]] if len(q)>3 else prtoid['__na__'],
                  entoid[q[4]] if len(q)>3 else entoid['__na__']] for q in train_quints]
        valid = [[entoid[q[0]],
                  prtoid[q[1]],
                  entoid[q[2]],
                  prtoid[q[3]] if len(q)>3 else prtoid['__na__'],
                  entoid[q[4]] if len(q)>3 else entoid['__na__']] for q in val_quints]
        test = [[entoid[q[0]],
                 prtoid[q[1]],
                 entoid[q[2]],
                 prtoid[q[3]] if len(q)>3 else prtoid['__na__'],
                 entoid[q[4]] if len(q)>3 else entoid['__na__']] for q in test_quints]

        return {"train": train, "valid": valid, "test": test, "n_entities": len(q_entities),
                "n_relations": len(q_predicates), 'e2id': entoid, 'r2id': prtoid}


def load_jf17k_statements(maxlen=15) -> Dict:
    PARSED_DIR = Path('./data/parsed_data/jf17k')

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

        train, valid, test = _pad_statements_(train, maxlen), \
                             _pad_statements_(valid, maxlen), \
                             _pad_statements_(test,maxlen)

        return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
                "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}




def load_dummy_dataset():
    """

    :return: a dummy dataset for a model to overfit
    """
    num_rows = 1000
    num_entities = 200
    num_relations = 20
    ds = [[]]


class DataManager(object):
    """ Give me your args I'll give you a path to load the dataset with my superawesome AI """

    @staticmethod
    def load(config: Union[dict, FancyDict]) -> Callable:
        """ Depends upon 'STATEMENT_LEN' and 'DATASET' """

        # Get the necessary dataset's things.
        assert config['DATASET'] in KNOWN_DATASETS, f"Dataset {config['DATASET']} is unknown."

        if config['DATASET'] == 'wd50k':
            if config['STATEMENT_LEN'] == 5:
                return load_wd50k_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd50k_triples
            else:
                return partial(load_wd50k_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wikipeople':
            if config['STATEMENT_LEN'] == 5:
                return load_wikipeople_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wikipeople_triples
            else:
                return partial(load_wikipeople_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd50k_100':
            if config['STATEMENT_LEN'] == 5:
                return load_wd50k_100_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd50k_100_triples
            else:
                return partial(load_wd50k_100_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd50k_100_33':
            if config['STATEMENT_LEN'] == 5:
                return load_wd50k_100_33_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd50k_100_33_triples
            else:
                return partial(load_wd50k_100_33_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd50k_100_66':
            if config['STATEMENT_LEN'] == 5:
                return load_wd50k_100_66_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd50k_100_66_triples
            else:
                return partial(load_wd50k_100_66_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd50k_33':
            if config['STATEMENT_LEN'] == 5:
                return load_wd50k_33_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd50k_33_triples
            else:
                return partial(load_wd50k_33_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd50k_66':
            if config['STATEMENT_LEN'] == 5:
                return load_wd50k_66_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_wd50k_66_triples
            else:
                return partial(load_wd50k_66_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'jf17k':
            if config['STATEMENT_LEN'] == 5:
                return load_jf17k_quints
            elif config['STATEMENT_LEN'] == 3:
                return load_jf17k_triples
            elif config['STATEMENT_LEN'] == -1:
                return partial(load_jf17k_statements, maxlen=config['MAX_QPAIRS'])

    @staticmethod
    def gather_missing_entities(data: List[list], n_ents: int, positions: List[int]) -> np.array:
        """

            Find the entities which aren't available from range(n_ents).
            Think inverse of gather_entities

        :param data: A list of triples/quints whatever
        :param n_ents: Int signifying total number of entities
        :param positions: the positions over which we intend to count these things.
        :return: np array of entities NOT appearing in range(n_ents)
        """

        appeared = np.zeros(n_ents, dtype=np.int)
        for datum in data:
            for pos in positions:
                appeared[datum[pos]] = 1

        # Return this removed from range(n_ents)
        return np.arange(n_ents)[appeared == 0]


    @staticmethod
    def get_graph_repr(raw: Union[List[List[int]], np.ndarray], config: dict) \
            -> Dict[str, np.ndarray]:
        """
            Decisions:
                We are NOT making inverse of qualifier relations. Those are just repeated.
                The normal triple relations are inverted.

            Pseudocode:
                for each of train, test, valid split
                    for each triple,
                        s, o -> edge_index
                        r -> edge_type
                        r_q1,... -> list of column vectors (np.arrs)
                        e_q1,... -> list of column vectors (np.arrs)
                    endfor
                endfor

                    create reverse relations in the existing stuff.

            TODO: Check if the data has repeats (should not).

            :param raw: [[s, p, o, qr1, qe1, qr2, qe3...], ..., [...]]
                (already have a max qualifier length padded data)
            :param config: the config dict
        """
        has_qualifiers: bool = config['STATEMENT_LEN'] != 3
        try:
            nr = config['NUM_RELATIONS']
        except KeyError:
            raise AssertionError("Function called too soon. Num relations not found.")

        edge_index, edge_type = np.zeros((2, len(raw) * 2), dtype='int32'), np.zeros((len(raw) * 2), dtype='int32')
        qual_rel = np.zeros(((len(raw[0]) - 3) // 2, len(raw) * 2), dtype='int32')
        qual_ent = np.zeros(((len(raw[0]) - 3) // 2, len(raw) * 2), dtype='int32')

        # Add actual data
        for i, data in enumerate(raw):
            edge_index[:, i] = [data[0], data[2]]
            edge_type[i] = data[1]

            # @TODO: add qualifiers
            if has_qualifiers:
                qual_rel[:, i] = data[3::2]
                qual_ent[:, i] = data[4::2]

        # Add inverses
        edge_index[1, len(raw):] = edge_index[0, :len(raw)]
        edge_index[0, len(raw):] = edge_index[1, :len(raw)]
        edge_type[len(raw):] = edge_type[:len(raw)] + nr

        if has_qualifiers:
            qual_rel[:, len(raw):] = qual_rel[:, :len(raw)]
            qual_ent[:, len(raw):] = qual_ent[:, :len(raw)]

            return {'edge_index': edge_index,
                    'edge_type': edge_type,
                    'qual_rel': qual_rel,
                    'qual_ent': qual_ent}
        else:
            return {'edge_index': edge_index,
                    'edge_type': edge_type}

    @staticmethod
    def get_alternative_graph_repr(raw: Union[List[List[int]], np.ndarray], config: dict) \
            -> Dict[str, np.ndarray]:
        """
        Decisions:

            Quals are represented differently here, i.e., more as a coo matrix
            s1 p1 o1 qr1 qe1 qr2 qe2    [edge index column 0]
            s2 p2 o2 qr3 qe3            [edge index column 1]

            edge index:
            [ [s1, s2],
              [o1, o2] ]

            edge type:
            [ p1, p2 ]

            quals will looks like
            [ [qr1, qr2, qr3],
              [qe1, qr2, qe3],
              [0  , 0  , 1  ]       <- obtained from the edge index columns

        :param raw: [[s, p, o, qr1, qe1, qr2, qe3...], ..., [...]]
            (already have a max qualifier length padded data)
        :param config: the config dict
        :return: output dict
        """
        has_qualifiers: bool = config['STATEMENT_LEN'] != 3
        try:
            nr = config['NUM_RELATIONS']
        except KeyError:
            raise AssertionError("Function called too soon. Num relations not found.")

        edge_index, edge_type = np.zeros((2, len(raw) * 2), dtype='int32'), np.zeros((len(raw) * 2), dtype='int32')
        # qual_rel = np.zeros(((len(raw[0]) - 3) // 2, len(raw) * 2), dtype='int32')
        # qual_ent = np.zeros(((len(raw[0]) - 3) // 2, len(raw) * 2), dtype='int32')
        qualifier_rel = []
        qualifier_ent = []
        qualifier_edge = []

        # Add actual data
        for i, data in enumerate(raw):
            edge_index[:, i] = [data[0], data[2]]
            edge_type[i] = data[1]

            # @TODO: add qualifiers
            if has_qualifiers:
                qual_rel = np.array(data[3::2])
                qual_ent = np.array(data[4::2])
                non_zero_rels = qual_rel[np.nonzero(qual_rel)]
                non_zero_ents = qual_ent[np.nonzero(qual_ent)]
                for j in range(non_zero_ents.shape[0]):
                    qualifier_rel.append(non_zero_rels[j])
                    qualifier_ent.append(non_zero_ents[j])
                    qualifier_edge.append(i)

        quals = np.stack((qualifier_rel, qualifier_ent, qualifier_edge), axis=0)
        num_triples = len(raw)

        # Add inverses
        edge_index[1, len(raw):] = edge_index[0, :len(raw)]
        edge_index[0, len(raw):] = edge_index[1, :len(raw)]
        edge_type[len(raw):] = edge_type[:len(raw)] + nr

        if has_qualifiers:
            full_quals = np.hstack((quals, quals))
            full_quals[2, quals.shape[1]:] = quals[2, :quals.shape[1]]  # TODO: might need to + num_triples

            return {'edge_index': edge_index,
                    'edge_type': edge_type,
                    'quals': full_quals}
        else:
            return {'edge_index': edge_index,
                    'edge_type': edge_type}

    @staticmethod
    def add_reciprocals(data: Union[List[List[int]], np.ndarray], config: dict) -> Union[List[List[int]], np.ndarray]:
        """

        :param data: original direct data
        :param config: config dict
        :return: data enriched with reverse triples
        """
        reci = []
        nr = config['NUM_RELATIONS']
        has_qualifiers: bool = config['STATEMENT_LEN'] != 3

        try:
            nr = config['NUM_RELATIONS']
        except KeyError:
            raise AssertionError("Function called too soon. Num relations not found.")

        for i, datum in enumerate(data):
            s, o = datum[0], datum[2]
            reci_r = datum[1] + nr

            reci_triple = [o, reci_r, s]

            if has_qualifiers:
                quals = datum[3:]
                reci_triple.extend(quals)

            reci.append(reci_triple)

        return reci


def count_stats(ds):
    import collections
    tr = ds['train']
    vl = ds['valid']
    ts = ds['test']
    ne = ds['n_entities']
    nr = ds['n_relations']
    print("Magic Mike!")
    print(f"The dataset: {len(tr)} training, {len(vl)} val, {len(ts)} test, {ne} entities, {nr} rels")
    if len(tr[0]) > 3:
        quals_train = len([x for x in tr if x[3] != 0])
        quals_val = len([x for x in vl if x[3] != 0])
        quals_test = len([x for x in ts if x[3] != 0])
        max_qlen = max([np.array(x).nonzero()[0][-1]+1 for x in tr+vl])
        distr = {k: v for k, v in collections.Counter(np.array(x).nonzero()[0][-1]+1 for x in tr+vl).items()}
        print(f"W/ quals - train: {quals_train}/{round(float(quals_train/len(tr)), 3)}")
        print(f"W/ quals - val: {quals_val}/{round(float(quals_val / len(vl)), 3)}")
        print(f"W/ quals - test: {quals_test}/{round(float(quals_test / len(ts)), 3)}")
        print(f"Max statement len w/ qual: {max_qlen}")
        print(distr)
    # id2e = {v:k for k,v in ds['e2id'].items()}
    # id2p = {v:k for k,v in ds['r2id'].items()}
    train_ents = set([item for x in tr for item in x[0::2]])
    train_rels = set([item for x in tr for item in x[1::2]])
    val_ents = set([item for x in vl for item in x[0::2]])
    val_rels = set([item for x in vl for item in x[1::2]])
    tv_ents = set([item for x in tr + vl for item in x[0::2]])
    tv_rels = set([item for x in tr + vl for item in x[1::2]])
    test_ents = set([item for x in ts for item in x[0::2]])
    test_rels = set([item for x in ts for item in x[1::2]])
    if len(tr[0]) > 3:
        qe = set([item for x in tr+vl+ts for item in x[4::2]])
        qr = set([item for x in tr+vl+ts for item in x[3::2]])
        all_ents = set([item for x in tr+vl+ts for item in [x[0],x[2]]])
        all_rels = set([x[1] for x in tr+vl+ts])
        print(f"Unique qual entities: {len(qe.difference(all_ents))}")
        print(f"Unique qual rels: {len(qr.difference(all_rels))}")

    dups_train = {k:v for k,v in collections.Counter(tuple(x) for x in tr).items() if v > 1}
    print("Duplicates in train: ", len(dups_train))
    dups_val = {k: v for k, v in collections.Counter(tuple(x) for x in vl).items() if v > 1}
    print("Duplicates in val: ", len(dups_val))
    dups_test = {k: v for k, v in collections.Counter(tuple(x) for x in ts).items() if v > 1}
    print("Duplicates in test: ", len(dups_test))
    print("-" * 10)

    ts_unique = val_ents.difference(train_ents)
    ts_unique_rel = val_rels.difference(train_rels)
    senseless_triples = []
    for x in vl:
        xe = set(x[0::2])
        xr = set(x[1::2])
        if len(xe.intersection(ts_unique)) > 0:
            senseless_triples.append(x)
            continue
        elif len(xr.intersection(ts_unique_rel)) > 0:
            senseless_triples.append(x)
            continue

    count = 0
    val_spos = set([(i[0], i[1], i[2]) for i in vl])
    for i in tr:
        main_triple = (i[0], i[1], i[2])
        if main_triple in val_spos:
            count += 1

    # senseless_triples = [x for x in vl if x[0] in ts_unique or x[2] in ts_unique]
    print(len(ts_unique), "/", ne, " entities are in val but not in train")
    print(len(ts_unique_rel), "/", nr, " rels are in val but not in train")
    print(f"Those entities and relations are used in {len(senseless_triples)} val triples")
    print("Leak triples in train wrt val: ", count, "/", len(tr))

    print("-"*10)
    ts_unique = test_ents.difference(tv_ents)
    ts_unique_rel = test_rels.difference(tv_rels)
    senseless_triples = []
    for x in ts:
        xe = set(x[0::2])
        xr = set(x[1::2])
        if len(xe.intersection(ts_unique)) > 0:
            senseless_triples.append(x)
            continue
        elif len(xr.intersection(ts_unique_rel)) > 0:
            senseless_triples.append(x)
            continue

    count = 0
    test_spos = set([(i[0], i[1], i[2]) for i in ts])
    for i in tr+vl:
        main_triple = (i[0], i[1], i[2])
        if main_triple in test_spos:
            count += 1
    # senseless_triples = [x for x in ts if x[0] in ts_unique or x[2] in ts_unique]
    print(len(ts_unique), "/", ne, " entities are in test but not in train+valid")
    print(len(ts_unique_rel), "/", nr, " rels are in test but not in train")
    print(f"Those entities and relations are used in {len(senseless_triples)} test triples")
    print("Leak triples in train+val wrt to test : ", count, "/", len(tr+vl))

    count = 0
    test_spos = set([(i[0], i[1], i[2]) for i in ts])
    test_opss = set([(i[2], i[1], i[0]) for i in ts])
    for i in tr + vl:
        main_triple = (i[0], i[1], i[2])
        rec_triple = (i[2], i[1], i[0])
        if main_triple in test_opss:
            count += 1
        if rec_triple in test_spos:
            count += 1
    # senseless_triples = [x for x in ts if x[0] in ts_unique or x[2] in ts_unique]
    print("Reverse triples in train+val wrt to test : ", count, "/", len(tr + vl))

    count = 0
    count_rev = 0
    trvl_spos = set([(i[0], i[1], i[2]) for i in tr+vl])
    for i in ts:
        main_triple = (i[0], i[1], i[2])
        rec_triple = (i[2], i[1], i[0])
        if main_triple in trvl_spos:
            count += 1
        if rec_triple in trvl_spos:
            count_rev += 1

    print("Triples in test that share the same base as in train+val : ", count, "/", len(ts))
    print("Triples in test that  are reverses of those in train+val : ", count_rev, "/", len(ts))




if __name__ == "__main__":
    ds = load_wikipeople_statements(maxlen=15)
    count_stats(ds)

