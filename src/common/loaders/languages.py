import numpy as np
import urllib.request
import os
import csv

from torch import nn
import torch

# pre-trained word embedding taken from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
embedding_urls = {
    'en': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec',
    'fr': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.vec',
    'es': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.vec',
    'it': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.it.vec',
    'de': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec',
    'pt': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.pt.vec',
    'zh': 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh.vec'
}

# dictionary taken from https://github.com/facebookresearch/MUSE
dictionary_urls = {
    'train': {
        'de-en': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.0-5000.txt',
        'en-de': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.0-5000.txt',
        'de-es': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/de-es.0-5000.txt',
        'es-de': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/es-de.0-5000.txt',
        'de-fr': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/de-fr.0-5000.txt',
        'fr-de': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-de.0-5000.txt',
        'de-it': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/de-it.0-5000.txt',
        'it-de': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/it-de.0-5000.txt',
        'de-pt': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/de-pt.0-5000.txt',
        'pt-de': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-de.0-5000.txt',
        'en-es': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.0-5000.txt',
        'es-en': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/es-en.0-5000.txt',
        'en-fr': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-fr.0-5000.txt',
        'fr-en': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-en.0-5000.txt',
        'en-it': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-it.0-5000.txt',
        'it-en': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/it-en.0-5000.txt',
        'en-pt': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-pt.0-5000.txt',
        'pt-en': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-en.0-5000.txt',
        'es-fr': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/es-fr.0-5000.txt',
        'fr-es': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-es.0-5000.txt',
        'es-it': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/es-it.0-5000.txt',
        'it-es': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/it-es.0-5000.txt',
        'es-pt': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/es-pt.0-5000.txt',
        'pt-es': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-es.0-5000.txt',
        'fr-it': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-it.0-5000.txt',
        'it-fr': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/it-fr.0-5000.txt',
        'fr-pt': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-pt.0-5000.txt',
        'pt-fr': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-fr.0-5000.txt',
        'itr-pt': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-pt.0-5000.txt',
        'pt-it': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-fr.0-5000.txt',
    },
    'test': {
        'de-en': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.5000-6500.txt',
        'en-de': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.5000-6500.txt',
        'de-es': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/de-es.5000-6500.txt',
        'es-de': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/es-de.5000-6500.txt',
        'de-fr': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/de-fr.5000-6500.txt',
        'fr-de': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-de.5000-6500.txt',
        'de-it': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/de-it.5000-6500.txt',
        'it-de': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/it-de.5000-6500.txt',
        'de-pt': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/de-pt.5000-6500.txt',
        'pt-de': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-de.5000-6500.txt',
        'en-es': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.5000-6500.txt',
        'es-en': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/es-en.5000-6500.txt',
        'en-fr': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-fr.5000-6500.txt',
        'fr-en': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-en.5000-6500.txt',
        'en-it': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-it.5000-6500.txt',
        'it-en': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/it-en.5000-6500.txt',
        'en-pt': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-pt.5000-6500.txt',
        'pt-en': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-en.5000-6500.txt',
        'es-fr': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/es-fr.5000-6500.txt',
        'fr-es': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-es.5000-6500.txt',
        'es-it': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/es-it.5000-6500.txt',
        'it-es': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/it-es.5000-6500.txt',
        'es-pt': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/es-pt.5000-6500.txt',
        'pt-es': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-es.5000-6500.txt',
        'fr-it': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-it.5000-6500.txt',
        'it-fr': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/it-fr.5000-6500.txt',
        'fr-pt': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-pt.5000-6500.txt',
        'pt-fr': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-fr.5000-6500.txt',
        'it-pt': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-pt.5000-6500.txt',
        'pt-it': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-fr.5000-6500.txt',
        'en-zh': 'https://dl.fbaipublicfiles.com/arrival/dictionaries/en-zh.5000-6500.txt',
    }
}


def download(url_loc, file_loc):
    urllib.request.urlretrieve(url_loc, file_loc)


def parse_dictionary(file_loc):
    with open(file_loc, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        data = csv.reader(f, delimiter=' ')
        return zip(*data)


def parse_dictionary(file_loc):
    with open(file_loc, 'r') as f:
        data = csv.reader(f, delimiter=' ')
        return list(zip(*data))


def parse_embeddings(file_loc, n_words):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    with open(file_loc, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        _, _emb_dim_file = map(int, next(f).rstrip().split())
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            word = word.lower()
            vect = np.fromstring(vect, sep=' ')
            if np.linalg.norm(vect) == 0:
                vect[0] = 0.01
            if word not in word2id:
                if not vect.shape == (_emb_dim_file,):
                    print('Invalid dimension')
                    continue
                assert vect.shape == (_emb_dim_file,), i
                word2id[word] = len(word2id)
                vectors.append(vect[None])
            if n_words and len(word2id) >= n_words:
                break

    assert len(word2id) == len(vectors)
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    print(f'Loaded {len(vectors)} pre-trained word embeddings.')
    return embeddings, word2id, _emb_dim_file


def embedding(root, src, n_words, most_freq_words):
    path = os.path.join(os.path.expanduser(root), 'embeddings')
    os.makedirs(path, exist_ok=True)
    src_loc = os.path.join(path, f'{src}.vec')
    if not os.path.isfile(src_loc):
        print(f'{src_loc} not found. Downloading.')
        download(embedding_urls[src], src_loc)
    print(f'Parsing embeddings from {src_loc}')
    emb, word2id, emb_size = parse_embeddings(src_loc, n_words)
    return emb, word2id, emb_size


def dictionary(root, src, tgt, split):
    path = os.path.join(os.path.expanduser(root), 'embeddings', split)
    os.makedirs(path, exist_ok=True)
    translation = f'{src}-{tgt}'
    src_loc = os.path.join(path, translation)
    if not os.path.isfile(src_loc):
        download(dictionary_urls[split][translation], src_loc)
    return parse_dictionary(src_loc)


def neighbors(root, word_embedding, batch_size):
    neigh = np.load(root)[1]
    while True:
        sidx = torch.LongTensor(batch_size).random_(neigh.shape[0])
        semb_idx = neigh[sidx][0]
        tidx = torch.LongTensor(batch_size).random_(1, neigh.shape[1])
        temb_idx = neigh[sidx][tidx]
        yield(word_embedding[semb_idx], word_embedding[temb_idx])


def sample_emb(emb, batch_size, most_freq_words, device):
    size = min(most_freq_words, len(emb)) if most_freq_words else len(emb)
    embed = nn.Embedding(len(emb), 300)
    embed.weight.data.copy_(emb)
    while True:
        idx = torch.LongTensor(batch_size).random_(size)
        yield(embed(idx).data)
