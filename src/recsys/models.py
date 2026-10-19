"""models"""
import numpy as np
import scipy.sparse as sp
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors


def _top_n_excluding_seen(scores: np.ndarray, X_rows: sp.csr_matrix, n: int) -> np.ndarray:
    """get top n, skip songs already heard"""
    seen_r, seen_c = X_rows.nonzero()
    scores[seen_r, seen_c] = -np.inf
    top = np.argpartition(-scores, kth=n - 1, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
    return np.take_along_axis(top, order, axis=1)


class MostPopular:
    """most popular songs"""

    name = "MostPopular"

    def fit(self, X: sp.csr_matrix):
        self.X = X
        self.popularity = np.asarray(X.sum(axis=0)).ravel()
        self.order = np.argsort(-self.popularity, kind="stable")
        return self

    def recommend(self, user_rows: np.ndarray, n: int) -> np.ndarray:
        # skip songs the user already heard
        out = np.empty((len(user_rows), n), dtype=np.int64)
        for i, r in enumerate(user_rows):
            seen = self.X.indices[self.X.indptr[r]:self.X.indptr[r + 1]]
            candidates = self.order[: n + len(seen)]
            out[i] = candidates[~np.isin(candidates, seen)][:n]
        return out


class ItemKNN:
    """item knn with cosine similarity"""

    name = "ItemKNN-cosine"

    def __init__(self, k: int = 100, n_jobs: int = 4, working_memory_mb: int = 256):
        self.k = k
        self.n_jobs = n_jobs
        self.working_memory_mb = working_memory_mb  # limit memory

    def fit(self, X: sp.csr_matrix):
        self.X = X
        items = X.T.tocsr()
        n_items = items.shape[0]
        k = min(self.k + 1, n_items)  # +1 for the item itself
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=self.n_jobs)
        nn.fit(items)
        with config_context(working_memory=self.working_memory_mb):
            dist, idx = nn.kneighbors(items)
        sim = (1.0 - dist).astype(np.float32)

        rows = np.repeat(np.arange(n_items), k)
        cols, vals = idx.ravel(), sim.ravel()
        keep = (cols != rows) & (vals > 0)
        S = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n_items, n_items))
        # similarity matrix
        self.S = S
        pop = np.asarray(X.sum(axis=0)).ravel()
        self.tiebreak = (1e-6 * pop / pop.max()).astype(np.float32)
        return self

    def recommend(self, user_rows: np.ndarray, n: int, batch: int = 256) -> np.ndarray:
        out = []
        for s in range(0, len(user_rows), batch):
            rows = user_rows[s:s + batch]
            Xr = self.X[rows]
            scores = (Xr @ self.S).toarray() + self.tiebreak
            out.append(_top_n_excluding_seen(scores, Xr, n))
        return np.vstack(out)
