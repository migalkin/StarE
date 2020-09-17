import pickle
from typing import List
import numpy as np
import random
from tqdm.autonotebook import tqdm

from utils.utils_mytorch import compute_mask
from utils.utils import *


class Corruption:
    """

        Used to efficiently corrupt given data.

        # Usage
        |-> If no filtering is needed (neg may exist in dataset)
            |-> simply init the class with n_ents, and call corrupt with pos data, and num corruptions
        |-> If filtering is needed (neg must not belong in dataset)
            |-> also pass all true data while init

        # Usage Example
        **no filtering**
        gold_data = np.random.randint(0, 1000, (50, 3))
        corrupter = Corruption(n=1000, position=[0, 2])
        corrupter.corrupt_one(gold_data, position=[0, 2])  # position overrides
        corrupter.precomute(true, neg_per_samples=1000) (although not much point of this)

        **with filtering**
        gold_data =  np.random.randint(0, 1000, (50, 3))
        corrupter = Corruption(n=1000, data=gold_data, position=[0, 2])
        corrupter.corrupt(gold_data, position=[0, 2]) # position overrides

        # Features
        - Precompute (not sure if we'll do this)
        - Filtering
        - Can return in pairwise fashion
            (p,n) pairs with repeated p's if needed
    """

    def __init__(self, n: int, position: list = None, excluding: np.array = None, gold_data: np.array = None,
                 debug: bool = False, caching: bool = False):
        """
            See class desc.

        :param n: range/set of possible negative entities
        :param position: which positions to corrupt in the eval data, which positions to hash
        :param gold_data: data to hash, to avoid entities
        :param debug: Syntactic sanity, more prints
        :param caching: never mind this for now.
        """
        self.n = n
        self.excluding = excluding
        self.position, self.debug = position, debug
        self.filtering = gold_data is not None
        self.hashes = self._index_(gold_data)

        self.caching = caching
        if caching:
            ...

    def _index_(self, data) -> Union[None, Dict[int, dict]]:
        """ Create hashes of trues"""
        if data is None:
            return None

        hashes = {pos: {} for pos in self.position}
        for datum in data:
            for _pos, _hash in hashes.items():
                _remainder = list(datum.copy())
                _real_val = _remainder.pop(_pos)

                _hash.setdefault(tuple(_remainder), []).append(_real_val)

        return hashes

    def corrupt_one_position(self, data: np.array, position: int) -> np.array:
        """
            Similar to corrupt_one but only generates negatives for a specific position.

        :param data: np.array of that which needs all forms of corruption
        :param position: the position for which we need to make the corruption
        :return: numpy array of corrupted things
        """
        assert position in self.position, "Invalid corruption position provided"

        # Get entities to exclude at this position
        key = list(data).copy()
        excluding = [key.pop(position)]

        if self.filtering:
            excluding += self.hashes[position][tuple(key)]

        excluding = np.array(excluding)
        excluding = np.sort(np.unique(np.concatenate((excluding, self.excluding))))
        including = np.arange(self.n) 
        entities = np.delete(including, excluding)

        corrupted = np.zeros((entities.shape[0], len(data)))
        corrupted[:, :] = data
        corrupted[:, position] = entities
        return corrupted

    def _get_entities_(self, bs: Union[int, np.array], excluding: Union[int, np.array] = None,
                       keys: np.array = None, data_hash: dict = None) -> np.array:
        """
            Step 1: Create random entities (n times)
            Step 2: If not filtering and excluding, a while loop to ensure all are replaced
            Step 3: If filtering, then we verify if the ent has not appeared in the dataset

        :param bs: number of things to inflect
        :param excluding:
            - int - don't have this entity
            - np.array - don't have these entities AT THESE POSITIONs
        :param keys: complete data used for filtering
        :param data_hash: the data hash we're using to do the filtering
        :return: (n,) entities
        """

        # Step 1: Choose an init set of entities excluding self.excluding
        entities = np.random.permutation(np.delete(np.arange(self.n), self.excluding))[:bs]

        # Step 2
        if excluding is not None and not self.filtering:

            # If excluding is single
            if type(excluding) in [int, float]:
                excluding = np.repeat(excluding, bs)

            if self.debug:
                repeats = 0

            while True:
                eq = entities == excluding
                if not eq.any():
                    # If they're completely dissimilar
                    break

                # Sample new entities
                new_entities = np.random.choice(np.delete(np.arange(self.n), self.excluding), int(np.sum(eq)))
                entities[eq] = new_entities

                if self.debug:
                    repeats += 1

        if self.debug:
            print(f"Corruption: The excluding loop went for {repeats} times.")

        # Step 3
        if self.filtering:
            # @TODO: later alligator
            raise NotImplementedError

        return entities

    def corrupt_batch(self, data: np.array, position=None) -> np.array:
        """
            For each positions in data, make inflections. n_infl = len(data) // len(position)
            NOTE: It computes a mask and does not inflect in positions not in the mask.

            Returns corrupted arr
        """

        position = self.position if position is None else position
        neg_data = np.copy(data)

        # Compute and trim mask
        mask = compute_mask(data).astype(np.int)
        skip_positions = list(set(range(data.shape[1])).difference(set(position)))
        mask[:, skip_positions] = 0

        # noisy_entities = npr.randint(0, data.shape[0], (data.shape[0],))
        noisy_entities = self._get_entities_(data.shape[0])
        for row_index in range(data.shape[0]):
            column_index = np.random.choice(np.argsort(-mask[row_index])[:np.sum(mask[row_index])])
            neg_data[row_index, column_index] = noisy_entities[row_index]

        return neg_data


if __name__ == "__main__":
    # Testing Corruption class
    true = np.random.randint(0, 20, (20000, 3))
    true[2] = true[1].copy()
    true[2][-1] = 99
    n = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 15, 18, 19])
    corruption = Corruption(n, position=[0, 2])
    one_pos = true[10]
    # print(one_pos)

    # neg = corruption.corrupt_one(one_pos)
    # print(neg[:10])
    # print(neg.shape)

    batch = np.random.permutation(true)[:5]
    print('------')
    print(batch.shape)
    n = corruption.corrupt_batch(batch)
    print(n.shape)
