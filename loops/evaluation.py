from functools import partial
from tqdm.autonotebook import tqdm
import types

# Local
from utils.utils import *
from utils.utils_mytorch import Timer


class EvaluationBenchGNNMultiClass:
    """
        Sampler which for each true triple,
            |-> compares an entity ar CORRUPTION_POSITITON with **all** possible entities, and reports metrics
    """

    def __init__(self,
                 data: Dict[str, Union[List[int], np.array]],
                 model: nn.Module,
                 n_ents: int,
                 excluding_entities: Union[int, np.array],
                 config: Dict,
                 bs: int,
                 metrics: list,
                 filtered: bool = False,
                 trim: float = None,
                 positions: List[int] = None):
        """
            :param data: {'index': list/iter of positive triples, 'eval': list/iter of positive triples}.
            Np array are appreciated
            :param model: the nn module we're testing
            :param excluding_entities: either an int (indicating n_entities), or a array of possible negative entities
            :param bs: anything under 256 is shooting yourself in the foot.
            :param metrics: a list of callable (from methods in this file) we call to get a metric
            :param filtered: if you want corrupted triples checked.
            :param trim: We could drop the 'eval' data, to speed things up
            :param positions: which positions should we inflect.
            """
        self.bs, self.filtered = bs, filtered
        self.model = model
        self.data_eval = data['eval']
        self.left_eval = self.data_eval[:(self.data_eval.shape[0] // 2), :]  # direct triples
        self.right_eval = self.data_eval[(self.data_eval.shape[0] // 2):, :]  # reci triples
        self.metrics = metrics
        self.excluding_entities = excluding_entities if config['ENT_POS_FILTERED'] else []

        # build an index of train/val/test data
        self.data = data
        self.config = config
        self.max_len_data = max(data['index'].shape[1], data['eval'].shape[1])
        self.corruption_positions = list(range(0, self.max_len_data, 2)) if not positions else positions
        self.build_index()

        if trim is not None:
            assert trim <= 1.0, "Trim ratio can not be more than 1.0"
            self.data_eval = np.random.permutation(self.data_eval)[:int(trim * len(self.data_eval))]

    def build_index(self):
        """
        the index is comprised of both INDEX and EVAL parts of the dataset
        essentially, merging train + val + test for true triple labels
        TODO think what to do with the index when we have >2 CORRUPTION POSITIONS
        :return: self.index with train/val/test entries
        """
        self.index = defaultdict(list)
        if len(self.corruption_positions) > 2:
            raise NotImplementedError

        for statement in np.concatenate((self.data['index'], self.data['eval']), axis=0):
            s, r, o, quals = statement[0], statement[1], statement[2], statement[3:] if self.data['eval'].shape[1] >= 3 else None
            reci_rel = r + self.config['NUM_RELATIONS']
            self.index[(s, r, *quals)].append(o) if self.config['SAMPLER_W_QUALIFIERS'] else self.index[(s, r)].append(o)
            # self.index[(o, reci_rel, *quals)].append(s) if self.config['SAMPLER_W_QUALIFIERS'] else self.index[(o, reci_rel)].append(s)

        for k, v in self.index.items():
            self.index[k] = list(set(v))



    def get_label(self, statements):
        """

        :param statements: array of shape (bs, seq_len) like (64, 43)
        :return: array of shape (bs, num_entities) like (64, 49113)

        for each line we search in the index for the correct label and assign 1 in the resulting vector
        """
        # statement shape for correct processing of the very last batch which size might be less than self.bs
        y = np.zeros((statements.shape[0], self.config['NUM_ENTITIES']), dtype=np.float32)


        for i, s in enumerate(statements):
            s, r, quals = s[0], s[1], s[3:] if self.data_eval.shape[1] > 3 else None
            lbls = self.index[(s, r, *quals)] if self.config['SAMPLER_W_QUALIFIERS'] else self.index[(s,r)]
            y[i, lbls] = 1.0

        return y

    def reset(self):
        """ Call when you wanna run again but not change hashes etc """
        raise NotImplementedError

    def _compute_metric_(self, scores: np.array) -> List[Union[float, np.float]]:
        """ See what metrics are to be computed, and compute them."""
        return [_metric(scores) for _metric in self.metrics]

    def _summarize_metrics_(self, accumulated_metrics: dict, eval_size: int) -> dict:
        """
            Aggregate metrics across time. Accepts np array of (len(self.data_eval), len(self.metrics))
        """
        # mean = np.mean(accumulated_metrics, axis=0)
        summary = {}

        for k, v in accumulated_metrics.items():
            summary[k] = v / float(eval_size) if k != 'count' else v

        return summary

    def _mean_metrics_(self, left: dict, right:dict) -> dict:
        # assume left and right have the same keys
        result = {}
        for k, v in left.items():
            result[k] = (left[k] + right[k]) / 2.0 if k != 'count' else v

        return result
    @staticmethod
    def summarize_run(summary: dict):
        """ Nicely print what just went down """
        print(f"This run over {summary['data_length']} datapoints took "
              f"%(time).3f min" % {'time': summary['time_taken'] / 60.0})
        print("---------\n")
        print('Object prediction results')
        for k, v in summary['left'].items():
            print(k, ':', "%(v).4f" % {'v': v})
        print("---------\n")
        print('Subject prediction results')
        for k, v in summary['right'].items():
            print(k, ':', "%(v).4f" % {'v': v})
        print("---------\n")
        print('Overall prediction results')
        for k, v in summary['metrics'].items():
            print(k, ':', "%(v).4f" % {'v': v})

    def compute(self, pred, obj, label, results):
        """
            Discard the predictions for all objects not in label (not currently evaluated)

        :param pred: a 2D bs, ne tensor containing bs distributions over entities
        :param obj: the actual objects being predicted
        :param label: a 2D bs, ne multi-hot tensor
            (where 1 -> the obj appeared in train/val/test split)
        :param ignored_entities: some entities we expect to not appear in s/o positions.
            can mention them here. Its a list like [2, 10, 3242344, ..., 69]
        :param results:
        :return:
        """
        ignored_entities = self.excluding_entities  # remove qualifier only entities if the flag says so

        b_range = torch.arange(pred.size()[0], device=self.config['DEVICE'])
        irrelevant = label.clone()
        irrelevant[b_range, obj] = 0            #
        irrelevant[:, ignored_entities] = 1     # Across batch, add 1 to ents never to be predicted
        pred[irrelevant.bool()] = -1000000
        '''
            At this point, pred has a -1000000 at all positions where
                label = 1 but it is not in objs.
                that is, if 
                    (0, 1, 5) and (0, 1, 6) were in the KG. 
                    And the current triple being evaluated is (0, 1, 9)
                    then pred[i_batch, 5] and pred[i_batch, 6] will be -100000 but
                        pred[i_batch, 9] will retain its values.
                        
            Then the problem is simply to find the rank of the indices we get from objs        
        '''
        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

        # results = {}
        ranks = ranks.float()
        results['count'] = torch.numel(ranks) + results.get('count', 0.0)
        results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
        results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
        for k in [0, 2, 4, 9]:
            results['hits_at {}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                'hits_at {}'.format(k + 1), 0.0)
        return results

    def run(self, *args, **kwargs):
        """
            Calling this iterates through different data points, obtains their labels,
            passes them to the model,
                collects the scores, computes the metrics, and reports them.
        """
        metrics = []
        self.model.eval()

        with Timer() as timer:
            with torch.no_grad():
                for position in self.corruption_positions:
                    metr = {}
                    if position == 0:
                        # evaluate "direct"
                        for i in range(self.left_eval.shape[0])[::self.bs]:
                            eval_batch_direct = self.left_eval[i: i + self.bs]
                            subs = torch.tensor(eval_batch_direct[:, 0], device=self.config['DEVICE'])
                            rels = torch.tensor(eval_batch_direct[:, 1], device=self.config['DEVICE'])
                            objs = torch.tensor(eval_batch_direct[:, 2], device=self.config['DEVICE'])
                            labels = torch.tensor(self.get_label(eval_batch_direct), device=self.config['DEVICE'])
                            if not self.config['SAMPLER_W_QUALIFIERS']:
                                scores = self.model.forward(subs, rels)
                            else:
                                quals = torch.tensor(eval_batch_direct[:, 3:], device=self.config['DEVICE'])
                                scores = self.model.forward(subs, rels, quals)
                            metr = self.compute(scores, objs, labels, metr)
                        left_metrics = self._summarize_metrics_(metr, len(self.left_eval))


                    elif position == 2:
                        # evaluate "reci"
                        for i in range(self.right_eval.shape[0])[::self.bs]:
                            eval_batch_reci = self.right_eval[i: i + self.bs]
                            subs = torch.tensor(eval_batch_reci[:, 0], device=self.config['DEVICE'])
                            rels = torch.tensor(eval_batch_reci[:, 1], device=self.config['DEVICE'])
                            objs = torch.tensor(eval_batch_reci[:, 2], device=self.config['DEVICE'])
                            labels = torch.tensor(self.get_label(eval_batch_reci), device=self.config['DEVICE'])
                            if not self.config['SAMPLER_W_QUALIFIERS']:
                                # eval_batch_reci = torch.cat((subs.unsqueeze(1), rels.unsqueeze(1), objs.unsqueeze(1)), dim=1)
                                scores = self.model.forward(subs, rels)
                            else:
                                quals = torch.tensor(eval_batch_reci[:, 3:], device=self.config['DEVICE'])
                                scores = self.model.forward(subs, rels, quals)
                            metr = self.compute(scores, objs, labels, metr)
                        right_metrics = self._summarize_metrics_(metr, len(self.right_eval))


        # Spruce up the summary with more information
        time_taken = timer.interval
        metrics = self._mean_metrics_(left_metrics, right_metrics)
        summary = {'metrics': metrics, 'time_taken': time_taken, 'data_length': len(self.data_eval),
                   'max_len_data': self.max_len_data, 'filtered': self.filtered, 'left': left_metrics, 'right': right_metrics}

        self.summarize_run(summary)

        return summary


def acc(scores: torch.Tensor) -> np.float:
    """ Accepts a (n, ) tensor """
    return (torch.argmin(scores, dim=0) == 0).float().detach().cpu().numpy().item()


def mrr(scores: torch.Tensor) -> np.float:
    """ Tested | Accepts one (n,) tensor """
    ranks = (torch.argsort(scores, dim=0) == 0).nonzero()[0]
    recirank = 1.0 / (ranks + 1).float()
    return recirank.detach().cpu().numpy().item()


def mr(scores: torch.Tensor) -> np.float:
    """ Tested | Accepts one (n,) tensor """
    ranks = (torch.argsort(scores, dim=0) == 0).nonzero()[0]
    ranks += 1
    return ranks.detach().cpu().numpy().item()


def hits_at(scores: torch.Tensor, k: int = 5) -> float:
    """ Tested | Accepts one (n,) tensor """
    rank = (torch.argsort(scores, dim=0) == 0).nonzero()[0] + 1
    if rank <= k:
        return 1.0
    else:
        return 0.0


def evaluate_pointwise(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> Union[int, float, bool]:
    """
        Given a pos and neg quint, how many times did the score for positive be more than score for negative

        :param pos_scores: scores corresponding to pos quints (bs, )
        :param neg_scores: scores corresponding to neg quints (bs, )
        :return accuracy (0d tensor)
    """
    return torch.mean((pos_scores < neg_scores).float()).item()


def evaluate_dataset(scores: torch.Tensor):
    """
        Compute score for `bs` set of [pos, neg, neg .....] quints.
        Assume pos is at the first position.


        :param scores: torch tensor of scores (bs,neg_samples+1)
        :returns (acc, mrr) both 1d tensors.
    """
    accuracy = (torch.argmin(scores, dim=1) == 0).float()
    ranks = (torch.argsort(scores, dim=1) == 0).nonzero()[:, 1]
    recirank = 1.0 / (ranks + 1).float()

    return accuracy.detach().cpu().numpy(), recirank.detach().cpu().numpy()


if __name__ == "__main__":
    print("smth")
