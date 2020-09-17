from utils.utils import *

class MultiClassSampler:
    """
        The sampler for the multi-class BCE training (instead of pointwise margin ranking)
        The output is a batch of shape (bs, num_entities)
        Each row contains 1s (or lbl-smth values) if the triple exists in the training set
        So given the triples (0, 0, 1), (0, 0, 4) the label vector will be [0, 1, 0, 0, 1]


    """
    def __init__(self, data: Union[np.array, list], n_entities: int,
                 lbl_smooth: float = 0.0, bs: int = 64, with_q: bool = False):
        """

        :param data: data as an array of statements of STATEMENT_LEN, e.g., [0,0,0] or [0,1,0,2,4]
        :param n_entities: total number of entities
        :param lbl_smooth: whether to apply label smoothing used later in the BCE loss
        :param bs: batch size
        :param with_q: whether indexing will consider qualifiers or not, default: FALSE
        """
        self.bs = bs
        self.data = data
        self.n_entities = n_entities
        self.lbl_smooth = lbl_smooth
        self.with_q = with_q

        self.build_index()
        self.keys = list(self.index.keys())
        self.shuffle()


    def shuffle(self):
        # npr.shuffle(self.data)
        npr.shuffle(self.keys)

    def build_index(self):
        self.index = defaultdict(list)

        for statement in self.data:
            s, r, quals = statement[0], statement[1], statement[3:] if self.data.shape[1] > 3 else None
            self.index[(s, r, *quals)].append(statement[2]) if self.with_q else self.index[(s, r)].append(statement[2])

        # remove duplicates in the objects list for convenience
        for k, v in self.index.items():
            self.index[k] = list(set(v))

    def reset(self, *ignore_args):
        """
            Reset the pointers of the iterators at the end of an epoch
        :return:
        """
        # do something
        self.i = 0
        self.shuffle()

        return self

    def get_label(self, statements):
        """

        :param statements: array of shape (bs, seq_len) like (64, 43)
        :return: array of shape (bs, num_entities) like (64, 49113)

        for each line we search in the index for the correct label and assign 1 in the resulting vector
        """
        # statement shape for correct processing of the very last batch which size might be less than self.bs
        y = np.zeros((statements.shape[0], self.n_entities), dtype=np.float32)


        for i, s in enumerate(statements):
            s, r, quals = s[0], s[1], s[2:] if self.data.shape[1] > 3 else None
            lbls = self.index[(s, r, *quals)] if self.with_q else self.index[(s,r)]
            y[i, lbls] = 1.0

        if self.lbl_smooth != 0.0:
            y = (1.0 - self.lbl_smooth)*y + (1.0 / self.n_entities)

        return y

    def __len__(self):
        # return self.data.shape[0] // self.bs
        return len(self.index) // self.bs

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        """
            Each time, take `bs` pos
        """
        if self.i >= len(self.keys)-1:  # otherwise batch norm will fail
            print("Should stop")
            raise StopIteration

        _statements = self.keys[self.i: min(self.i + self.bs, len(self.keys))]
        _main = np.array([list(x) for x in _statements])
        _labels = self.get_label(_main)
        self.i = min(self.i + self.bs, len(self.keys))
        return _main, _labels

