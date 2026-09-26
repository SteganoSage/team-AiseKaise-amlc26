"""Parallel hashed TF-IDF.

sklearn's TfidfVectorizer tokenises in pure Python on one core, which dominated
blocking time on ~5M strings per country. Here tokenisation is a stateless
HashingVectorizer run in worker processes; document frequencies, the max_df /
min_df cut, IDF weighting and L2 normalisation are then applied with sparse
matrix operations. Output matches sklearn's smooth-IDF, sublinear-TF scheme up
to (negligible) hash collisions in 2**22 buckets.
"""
import os
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

N_FEATURES = 2 ** 22
_CHUNK = 200_000


def _hash_chunk(args):
    texts, cfg = args
    hv = HashingVectorizer(n_features=N_FEATURES, alternate_sign=False, norm=None,
                           dtype=np.float32, lowercase=False, **cfg)
    return hv.transform(texts)


def hashed_counts(texts, cfg, pool=None):
    """Raw term counts (CSR) for a list of strings."""
    tasks = [(texts[i:i + _CHUNK], cfg) for i in range(0, len(texts), _CHUNK)]
    if pool is None:
        with Pool(max(1, (os.cpu_count() or 2) - 1)) as p:
            parts = p.map(_hash_chunk, tasks)
    else:
        parts = pool.map(_hash_chunk, tasks)
    return sp.vstack(parts, format="csr") if parts else sp.csr_matrix((0, N_FEATURES), dtype=np.float32)


def tfidf_pair(q_texts, i_texts, cfg, max_df=None, min_df=2, pool=None):
    """TF-IDF for query and index texts with IDF fitted on both together.

    max_df / min_df are absolute document counts over the combined corpus.
    Returns (Q, I) as L2-normalised float32 CSR matrices with the same columns.
    """
    q = hashed_counts(q_texts, cfg, pool)
    m = hashed_counts(i_texts, cfg, pool)
    n_docs = q.shape[0] + m.shape[0]
    df = np.bincount(q.indices, minlength=N_FEATURES) + np.bincount(m.indices, minlength=N_FEATURES)
    idf = (np.log((1 + n_docs) / (1 + df)) + 1).astype(np.float32)
    keep = df >= min_df
    if max_df is not None:
        keep &= df <= max_df
    weight = sp.diags(np.where(keep, idf, 0).astype(np.float32))
    out = []
    for x in (q, m):
        x.data = 1 + np.log(x.data)
        x = x @ weight
        x.eliminate_zeros()
        out.append(normalize(x, copy=False).astype(np.float32))
    return out[0], out[1]
