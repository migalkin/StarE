
<h1 align="center">
  StarE
</h1>

<h4 align="center">Message Passing for Hyper-Relational Knowledge Graph.</h4>


<p align="center">
 
<a href="https://doi.org/10.5281/zenodo.4036498"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4036498.svg" alt="DOI"></a>
<a href="https://github.com/migalkin/StarE/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg">
</p>

<h2 align="center">
  Overview of StarE
  <img align="center"  src="./architecture.png" alt="...">
</h2>

StarE encodes hyper-relational fact by first passing Qualifier pairs through a composition function <img src="https://render.githubusercontent.com/render/math?math=\phi_q"> and then summed and transformed by <img src="https://render.githubusercontent.com/render/math?math=\mathbf{W}_q">.
 The resulting vector is then merged via <img src="https://render.githubusercontent.com/render/math?math=\gamma">, and <img src="https://render.githubusercontent.com/render/math?math=\phi_r"> with the relation and object vector, respectively. Finally, node **Q937** aggregates messages from this and other hyper-relational edges. Please refer to the paper for details.

## Requirements
* Python 3.7
* PyTorch 1.5.1
* torch-geometric 1.6.1
* torch-scatter 2.0.5
* tqdm
* wandb

Create a new conda environment and execute `setup.sh`.
Alternatively
```
pip install -r requirements.txt
```

## WD50K Dataset
The dataset can be found in `data/clean/wd50k`.
Its derivatives can be found there as well:
* `wd50k_33` - approx 33% of statements have qualifiers
* `wd50k_66` - approx 66% of statements have qualifiers
* `wd50k_100` - 100% of statements have qualifiers

## Running Experiments

### Available models
Specified as `MODEL_NAME` in the running script
* `stare_transformer` - main model StarE (H) + Transformer (H) [default]
* `stare_stats_baseline` - baseline model Transformer (H)
* `stare_trans_baseline` - baseline model Transformer (T)

### Datasets
Specified as `DATASET` in the running script
* `jf17k`
* `wikipeople`
* `wd50k` [default]
* `wd50k_33` 
* `wd50k_66`
* `wd50k_100`

### Starting training and evaluation
It is advised to run experiments on a GPU otherwise training might take long.
Use `DEVICE cuda` to turn on GPU support, default is `cpu`.
Don't forget to specify `CUDA_VISIBLE_DEVICES` before `python` if you use `cuda`

Three parameters control triple/hyper-relational nature and max fact length:
* `STATEMENT_LEN`: `-1` for hyper-relational [default], `3` for triples
* `MAX_QPAIRS`: max fact length (3+2*quals), e.g., `15` denotes a fact with 5 qualifiers `3+2*5=15`.
`15` is default for `wd50k` datasets and `jf17k`, set `7` for wikipeople, set `3` for triples (in combination with `STATEMENT_LEN 3`) 
* `SAMPLER_W_QUALIFIERS`: `True` for hyper-relational models [default], `False` for triple-based models only 

The following scripts will train StarE (H) + Transformer (H) for 400 epochs and evaluate on the test set:

* StarE (H) + Transformer (H)
```
python run.py DATASET wd50k
```  
* StarE (H) + Transformer (H) with a GPU.
```
CUDA_VISIBLE_DEVICES=0 python run.py DEVICE cuda DATASET wd50k
``` 
*  You can adjust the dataset with a higher ratio of quals by changing `DATASET` with the available above names
```
python run.py DATASET wd50k_33
```
* On JF17K
```
python run.py DATASET jf17k CLEANED_DATASET False
```
* On WikiPeople
```
python run.py DATASET wikipeople CLEANED_DATASET False MAX_QPAIRS 7 EPOCHS 500
```

Triple-based models can be started with this basic set of params:
```
python run.py DATASET wd50k STATEMENT_LEN 3 MAX_QPAIRS 3 SAMPLER_W_QUALIFIERS False
```

More hyperparams are available in the `CONFIG` dictionary in the `run.py`.

If you want to adjust StarE encoder params prepend `GCN_` to the params in the `STAREARGS` dict, e.g., 
```
python run.py DATASET wd50k GCN_GCN_DIM 80 GCN_QUAL_AGGREGATE concat
```
will construct StarE with hidden dim of 80 and concat as `gamma` function from the paper.



#### When using the dataset please cite:

```
@inproceedings{StarE,
  title={Message Passing for Hyper-Relational Knowledge Graphs},
  author={Galkin, Mikhail and Trivedi, Priyansh and Maheshwari, Gaurav and Usbeck, Ricardo and Lehmann, Jens},
  booktitle={EMNLP},
  year={2020}
}
```

For any further questions, please contact:  ```mikhail.galkin@iais.fraunhofer.de```