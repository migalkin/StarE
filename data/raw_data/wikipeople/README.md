## WikiPeople: An n-ary relational dataset about people derived from Wikidata


WikiPeople was constructed as follows:
+ The [Wikidata dump](https://archive.org/details/wikibase-wikidatawiki-20171120) was downloaded and the facts concerning entities of type `human` were extracted.
+ Then, these facts were denoised. For example, facts containing element related to image were filtered out, and facts containing element in {*unknown value*, *no values*} were removed.
+ Subsequently, the subsets of elements which have at least 30 mentions were selected. And the facts related to these elements were kept. Further, each fact was parsed into a set of its role-value pairs.
+ The remaining facts were randomly split into training set, validation set and test set by a proportion of 80%:10%:10%.


The statistics of WikiPeople are displayed as follows, where #Train, #Valid and #Test are the sizes of the training set, the validation set and the test set, respectively.

|  | Binary | N-ary | Overall |
| :-: | :-: | :-: | :-: |
| #Train | 270,179 | 35,546 | 305,725 |
| #Valid | 33,845 | 4,378 | 38,223 |
| #Test | 33,890 | 4,391 | 38,281 |


The training facts, validation facts and test facts stored in the files `n-ary_train.json`, `n-ary_valid.json` and `n-ary_test.json`, respectively, are in the same format. Each line therein is a set of ("role id": "value id/list of value ids") and the arity information in form of ("N": arity), corresponding to a fact in Wikidata about a certain person. Note that all the ids, except the ones that end with "_h" or "_t", are directly adopted from Wikidata. The two types of ids ending with "_h" or "_t" are defined by WikiPeople.

### A fact example of WikiPeople
Take `Line 37714` in `n-ary_valid.json` for example:

    {
      "P166_h": "Q7186", 
      "P166_t": "Q38104", 
      "N": 5, 
      "P585": ["+1903-01-01T00:00:00Z#0#0#0#9#http://www.wikidata.org/entity/Q1985727"], 
      "P1706": ["Q41269", "Q37463"]
    }


This example corresponds to the fact in [Wikidata](https://www.wikidata.org/wiki/Q7186): ***Marie Curie*** received ***Nobel Prize in Physics*** in ***1903***, together with ***Henri Becquerel*** and ***Pierre Curie***.


The detailed description of `Line 37714` is as follows (items newly defined or introdued by WikiPeople are in italics and underlined):

| Item | Description | 
| :-: | - |
| P166 | The id of the relation "award received" |
| <ins>*P166_h*</ins> | The id of the subject role of "award received" |
| Q7186 | The id of the value "Marie Curie" |
| <ins>*P166_t*</ins> | The id of the object role of "award received" | 
| Q38104 | The id of the value "Nobel Prize in Physics" |
| <ins>*"N":5*</ins> | The arity of this fact is 5 |
| P585 | The id of the role "point in time" |
| +1903-01-01T00:00:00Z#0#0#0#9<br>#http://www.wikidata.org/entity/Q1985727 | The id of the value "1903" |
| P1706 | The id of the role "together with" |
| Q41269 | The id of the value "Henri Becquerel" |
| Q37463 | The id of the value "Pierre Curie" |


### When using the dataset, please cite:

    @inproceedings{NaLP,
      title={Link prediction on n-ary relational data},
      author={Guan, Saiping and Jin, Xiaolong and Wang, Yuanzhuo and Cheng, Xueqi},
      booktitle={Proceedings of the 28th International Conference on World Wide Web (WWW'19)},
      year={2019},
      pages={583--593}
    }
    
For any questions, please contact guansaiping@ict.ac.cn or jinxiaolong@ict.ac.cn, or open an issue.
