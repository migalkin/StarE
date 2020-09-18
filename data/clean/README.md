# WD50k dataset: An hyper-relational dataset derived from Wikidata statements

The dataset is constructed by the following procedure based on the [Wikidata RDF dump](https://dumps.wikimedia.org/wikidatawiki/20190801/) of August 2019:

-  A set of seed nodes corresponding to entities from FB15K-237 having a direct mapping in Wikidata (_P646_ "Freebase ID") is extracted from the dump.
-  For each seed node, all statements whose _main_ object and _qualifier_ values corresponding to _wikibase:Item_ are extracted from the dump.
-  All literals are filtered out from the qualifiers of the above obtained statements.
-  All the entities from the dataset which have less than two mentions are dropped. The statements corresponding to the dropped entities are also dropped.
-  The remaining statements are randomly split into the train, test, and validation sets.
-  All statements from train and validation sets are removed which share the same main triple __(s,p,o)__ with test statements.
-  WD50k_33, WD50k_66, WD50k_100 are then sampled from the above statements. Here 33, 66, 100 represents the amount of hyper-relational facts (statements with qualifiers) in the dataset.


The table below provides some basic statistics of our dataset and its three further variations:

| Dataset     | Statements | w/Quals (%)    | Entities | Relations | E only in Quals | R only in Quals | Train   | Valid  | Test   |
|-------------|------------|----------------|----------|-----------|-----------------|-----------------|---------|--------|--------|
| WD50K       | 236,507    | 32,167 (13.6%) | 47,156   | 532       | 5460            | 45              | 166,435 | 23,913 | 46,159 |
| WD50K (33)  | 102,107    | 31,866 (31.2%) | 38,124   | 475       | 6463            | 47              |  73,406 | 10,668 | 18,133 |
| WD50K (66)  |  49,167    | 31,696 (64.5%) | 27,347   | 494       | 7167            | 53              |  35,968 |  5,154 |  8,045 |
| WD50K (100) |  31,314    | 31,314 (100%)  | 18,792   | 279       | 7862            | 75              |  22,738 |  3,279 |  5,297 |


Each dataset i.e. wd50k and its derivatives (for example wd_50k) consists of two folders 

- __statements__ which corresponds to dataset with qualifiers. 
- __triples__ which corresponds to dataset where all qualifiers have been removed. For example a fact in _statements_ __(s, r, o, {(qr_1,  qv_1), (qr_2, qv_2)})__ is reduced to __(s, r, o,)__ in _triples_.

These folders have ```train.txt```, ```test.txt``` and ```valid.txt``` corresponding to train, test, and valid splits. Each line in the text file represents a fact/statement in the format __s, r, o, qr_1,  qv_1, qr_2, qv_2 ...__. The first three elements (s,r,o) represents the main triple, and the remaining part is the qualifier information. Note that the fact might not contain qualifier information (qr_1,  qv_1, qr_2, qv_2 ...). Below are a few examples of the dataset:

```
Q515632,P1196,Q3739104
Q219546,P1411,Q103916,P805,Q369706,P1686,Q3241699
Q131074,P166,Q487136,P805,Q458646,P1346,Q630767,P1346,Q15840165
Q965,P530,Q117,P805,Q2564434
Q825807,P1889,Q502273
```

In the above snippet for line ```Q219546,P1411,Q103916,P805,Q369706,P1686,Q3241699```

| Item | Description |
| ------ | ------ |
| Q219546 | subject (s) |
| P1411 | relation (r) |
| Q103916 | object (o) |
| P805 | qualifier relation 1 (qr_1) |
| Q369706 | qualifier entity 1 (qe_1)|
| P1686 | qualifier relation 2 (qr_2) |
| Q3241699 | qualifier entity 2 (qe_2) |

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
