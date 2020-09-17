from functools import partial
from typing import List, Union, Dict, Callable
import numpy as np
from utils.utils_mytorch import FancyDict
from utils.utils import KNOWN_DATASETS

from .load import load_jf17k_statements, load_jf17k_quints, load_jf17k_triples, \
    load_wikipeople_statements, load_wikipeople_quints, load_wikipeople_triples, \
    load_wd50k_statements, load_wd50k_quints, load_wd50k_triples, \
    load_wd50k_100_statements, load_wd50k_100_quints, load_wd50k_100_triples, \
    load_wd50k_66_statements, load_wd50k_66_quints, load_wd50k_66_triples, \
    load_wd50k_33_statements, load_wd50k_33_quints, load_wd50k_33_triples

from .clean_datasets import load_clean_wikipeople_statements, load_clean_jf17k_statements, load_clean_wd50k

class DataManager(object):
    """ Give me your args I'll give you a path to load the dataset with my superawesome AI """

    @staticmethod
    def load(config: Union[dict, FancyDict]) -> Callable:
        """ Depends upon 'STATEMENT_LEN' and 'DATASET' """

        # Get the necessary dataset's things.
        assert config['DATASET'] in KNOWN_DATASETS, f"Dataset {config['DATASET']} is unknown."

        if config['DATASET'] == 'wd50k':
            if config['STATEMENT_LEN'] == 5:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k", subtype="quints")
                else:
                    return load_wd50k_quints
            elif config['STATEMENT_LEN'] == 3:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k", subtype="triples")
                else:
                    return load_wd50k_triples
            else:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k", subtype="statements", maxlen=config['MAX_QPAIRS'])
                else:
                    return partial(load_wd50k_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wikipeople':
            if config['STATEMENT_LEN'] == 5:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wikipeople_statements, subtype="quints")
                else:
                    return load_wikipeople_quints
            elif config['STATEMENT_LEN'] == 3:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wikipeople_statements, subtype="triples")
                else:
                    return load_wikipeople_triples
            else:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wikipeople_statements, subtype="statements", maxlen=config['MAX_QPAIRS'])
                else:
                    return partial(load_wikipeople_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd50k_100':
            if config['STATEMENT_LEN'] == 5:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k_100", subtype="quints")
                else:
                    return load_wd50k_100_quints
            elif config['STATEMENT_LEN'] == 3:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k_100", subtype="triples")
                else:
                    return load_wd50k_100_triples
            else:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k_100", subtype="statements", maxlen=config['MAX_QPAIRS'])
                else:
                    return partial(load_wd50k_100_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd50k_33':
            if config['STATEMENT_LEN'] == 5:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k_33", subtype="quints")
                else:
                    return load_wd50k_33_quints
            elif config['STATEMENT_LEN'] == 3:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k_33", subtype="triples")
                else:
                    return load_wd50k_33_triples
            else:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k_33", subtype="statements", maxlen=config['MAX_QPAIRS'])
                else:
                    return partial(load_wd50k_33_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'wd50k_66':
            if config['STATEMENT_LEN'] == 5:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k_66", subtype="quints")
                else:
                    return load_wd50k_66_quints
            elif config['STATEMENT_LEN'] == 3:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k_66", subtype="triples")
                else:
                    return load_wd50k_66_triples
            else:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_wd50k, name="wd50k_66", subtype="statements", maxlen=config['MAX_QPAIRS'])
                else:
                    return partial(load_wd50k_66_statements, maxlen=config['MAX_QPAIRS'])
        elif config['DATASET'] == 'jf17k':
            if config['STATEMENT_LEN'] == 5:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_jf17k_statements, subtype="quints")
                else:
                    return load_jf17k_quints
            elif config['STATEMENT_LEN'] == 3:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_jf17k_statements, subtype="triples")
                else:
                    return load_jf17k_triples
            elif config['STATEMENT_LEN'] == -1:
                if config['CLEANED_DATASET']:
                    return partial(load_clean_jf17k_statements, subtype="statements", maxlen=config['MAX_QPAIRS'])
                else:
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