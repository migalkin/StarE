import torch
import sys
from pathlib import Path
from functools import lru_cache
from data_loaders import clean_datasets
from SPARQLWrapper import SPARQLWrapper, JSON
from gensim.models import KeyedVectors as kv


def create_file(emb: dict, emb_size, file_name):
    final_strings = [[str(len(emb) - 1)] + [str(emb_size)] + ['\n']]

    for key, value in emb.items():
        if key != 0:
            final_strings.append([str(key)] +
                                 [str(v) for v in value] + ['\n'])

    with open(file_name, 'w') as file:
        for f in final_strings:
            file.write(" ".join(f))


def create_embed_model(MODEL_PATH, CUSTOM_MODEL_PATH):
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    try:
        ent_w = model['init_embed']
    except:
        ent_w = model['entities']

    data = clean_datasets.load_clean_wd50k(name='wd50k', subtype='statements', maxlen=15)
    assert data['n_entities'] == ent_w.shape[0]

    # Creating dictionary.
    ent = {}
    ent_embsize = 0
    for i, weight in enumerate(ent_w):
        ent[i] = weight.numpy()
        ent_embsize = len(weight)
    file_name = Path(CUSTOM_MODEL_PATH) / Path('entity_embed')
    create_file(ent, ent_embsize, file_name)
    ent_model = kv.load_word2vec_format(file_name)
    return ent_model, data




@lru_cache(128)
def get_label(entity):
    endpoint_url = "https://query.wikidata.org/sparql"
    query = """SELECT ?itemLabel 
    WHERE 
    { wd:%(ENT)s rdfs:label ?itemLabel. FILTER (lang(?itemLabel)="en") }"""

    query = query % {'ENT': entity}

    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    return results["results"]["bindings"][0]['itemLabel']['value']


def get_nearest_neighbour(ent_model, ent_id, id2e, k=5):
    nearest = ent_model.similar_by_word(str(ent_id))
    final = []
    for n in nearest[:k]:
        e = id2e[int(n[0])]
        label = get_label(e)
        final.append((n[0], n[1], e, label))

    return final


def id2label(ent_id, id2e):
    e = id2e[int(ent_id)]
    return e, get_label(e)


def get_difference(ent, data, ent_model_1, id2e_model_1, ent_model_2, id2e_model_2):
    ent_id = data['e2id'][ent]
    print(f"given entity label {id2label(ent_id, id2e_model_1)}")
    print("for star_s")
    print(get_nearest_neighbour(ent_model_1, ent_id, id2e_model_1))
    print("for traf_s")
    print(get_nearest_neighbour(ent_model_2, ent_id, id2e_model_2))


if __name__ == '__main__':

    ent_model_star_s, data_star_s = create_embed_model('data/models/stare_s/model.torch', 'data/models/stare_s/')
    id2e_star_s = {v: k for k, v in data_star_s['e2id'].items()}

    ent_model_trf_s, data_trf_s = create_embed_model('data/models/trf_s/model.torch', 'data/models/trf_s/')
    id2e_trf_s = {v: k for k, v in data_trf_s['e2id'].items()}

    get_difference('Q849697', data_star_s, ent_model_star_s, id2e_star_s, ent_model_trf_s, id2e_trf_s)