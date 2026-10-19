"""Recommenders. Both share fit(X) / recommend(user_rows, n) and never return songs a user already heard."""
import numpy as np
import scipy.sparse as sp
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors


def _top_n_excluding_seen(scores: np.ndarray, X_rows: sp.csr_matrix, n: int) -> np.ndarray:
    """Row-wise top-n column indices of `scores`, with each user's training items masked out."""
    seen_r, seen_c = X_rows.nonzero()  # note: modifies `scores` in place
    scores[seen_r, seen_c] = -np.inf
    top = np.argpartition(-scores, kth=n - 1, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
    return np.take_along_axis(top, order, axis=1)


class MostPopular:
    """Naive benchmark: rank songs by number of distinct training listeners; same list for everyone."""

    name = "MostPopular"

    def fit(self, X: sp.csr_matrix):
        self.X = X
        self.popularity = np.asarray(X.sum(axis=0)).ravel()
        self.order = np.argsort(-self.popularity, kind="stable")
        return self

    def recommend(self, user_rows: np.ndarray, n: int) -> np.ndarray:
        # The ranking is global, so walk the popularity list and skip each user's seen songs.
        out = np.empty((len(user_rows), n), dtype=np.int64)
        for i, r in enumerate(user_rows):
            seen = self.X.indices[self.X.indptr[r]:self.X.indptr[r + 1]]
            candidates = self.order[: n + len(seen)]
            out[i] = candidates[~np.isin(candidates, seen)][:n]
        return out


class ItemKNN:
    """Course baseline: item-based k-nearest-neighbours with cosine similarity.

    Each song is a vector of its listeners. For every song we keep its `k` most cosine-similar songs
    (sklearn NearestNeighbors, brute force). A user's score for a candidate song is the sum of its
    similarities to the songs they already listened to. Ties (e.g. users whose songs have no neighbours
    among the candidates) are broken by popularity via a tiny epsilon so lists are always full.
    """

    name = "ItemKNN-cosine"

    def __init__(self, k: int = 100, n_jobs: int = 4, working_memory_mb: int = 256):
        self.k = k
        self.n_jobs = n_jobs
        self.working_memory_mb = working_memory_mb  # caps the dense distance chunks (8 GB laptops)

    def fit(self, X: sp.csr_matrix):
        self.X = X
        items = X.T.tocsr()  # items x users
        n_items = items.shape[0]
        k = min(self.k + 1, n_items)  # +1 because each item is its own nearest neighbour
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=self.n_jobs)
        nn.fit(items)
        with config_context(working_memory=self.working_memory_mb):
            dist, idx = nn.kneighbors(items)
        sim = (1.0 - dist).astype(np.float32)

        rows = np.repeat(np.arange(n_items), k)
        cols, vals = idx.ravel(), sim.ravel()
        keep = (cols != rows) & (vals > 0)
        S = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n_items, n_items))
        # Row i holds the neighbours of item i; scoring is X @ S (sum of similarities to the user's items).
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
