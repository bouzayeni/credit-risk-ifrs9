"""Weight-of-evidence (WoE) binning for scorecard features.

Conventions (standard in credit scoring):

- target y = 1 means default ("bad"), 0 means "good".
- WoE of a bin = ln(share of all goods in the bin / share of all bads in the bin). Positive
  WoE = safer than average. 0.5 is added to counts so empty cells stay finite.
- IV (information value) = sum over bins of (share of goods - share of bads) * WoE.
  Rough reading: < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, > 0.3 strong.

Numeric features: start from quantile bins, then merge neighbours until the default rate is
monotonic across bins (pool-adjacent-violators) and every bin holds at least MIN_BIN_SHARE of
the data. Monotonic bins make the scorecard explainable ("more recent inquiries, fewer
points"). Missing values get their own bin.

Categorical features: one bin per level; levels under MIN_LEVEL_SHARE are pooled into
one rare-levels bin (OTHER). Unseen levels at scoring time also go to "OTHER".
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INITIAL_BINS = 20
MIN_BIN_SHARE = 0.05
MIN_LEVEL_SHARE = 0.01
SMOOTHING = 0.5
MISSING = "MISSING"
OTHER = "RARE (<1% each)"


def woe_iv(bads: np.ndarray, goods: np.ndarray) -> tuple[np.ndarray, float]:
    bads = np.asarray(bads, dtype=float) + SMOOTHING
    goods = np.asarray(goods, dtype=float) + SMOOTHING
    dist_bad, dist_good = bads / bads.sum(), goods / goods.sum()
    woe = np.log(dist_good / dist_bad)
    return woe, float(((dist_good - dist_bad) * woe).sum())


def monotonic_merge(edges: list[float], bads: list[float], counts: list[float],
                    increasing: bool) -> list[float]:  # fmt: skip
    """Pool adjacent violators: merge neighbouring bins until bad rates are monotonic.

    edges are the upper bounds of each bin (the last is +inf). Returns the kept edges.
    """
    blocks: list[list[float]] = []  # [upper edge, bads, count]
    for edge, bad, n in zip(edges, bads, counts, strict=True):
        blocks.append([edge, bad, n])
        while len(blocks) > 1:
            (_, b1, n1), (e2, b2, n2) = blocks[-2], blocks[-1]
            rate1, rate2 = b1 / n1, b2 / n2
            violates = rate2 < rate1 if increasing else rate2 > rate1
            if not violates:
                break
            blocks[-2:] = [[e2, b1 + b2, n1 + n2]]
    return [b[0] for b in blocks]


def merge_small(edges: list[float], bads: list[float], counts: list[float],
                min_count: float) -> list[float]:  # fmt: skip
    """Merge the smallest bin into the neighbour with the closest bad rate until all are big
    enough. Merging neighbours of a monotonic sequence keeps it monotonic."""
    blocks = [[e, b, n] for e, b, n in zip(edges, bads, counts, strict=True)]
    while len(blocks) > 1:
        i = min(range(len(blocks)), key=lambda k: blocks[k][2])
        if blocks[i][2] >= min_count:
            break
        rate = blocks[i][1] / blocks[i][2]
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(blocks)]
        j = min(neighbours, key=lambda k: abs(blocks[k][1] / blocks[k][2] - rate))
        lo, hi = sorted((i, j))
        blocks[lo:hi + 1] = [[blocks[hi][0], blocks[lo][1] + blocks[hi][1],
                              blocks[lo][2] + blocks[hi][2]]]  # fmt: skip
    return [b[0] for b in blocks]


@dataclass
class FeatureBins:
    """Fitted bins for one feature: labels, WoE per bin and how to assign values."""

    name: str
    kind: str  # "numeric" or "categorical"
    labels: list[str]
    woe: list[float]
    bads: list[int]
    counts: list[int]
    iv: float
    edges: list[float] = field(default_factory=list)  # numeric: upper bounds, last = inf
    levels: dict[str, str] = field(default_factory=dict)  # categorical: level -> label
    increasing: bool | None = None  # numeric: does the bad rate rise with the value?

    def assign(self, x: pd.Series) -> np.ndarray:
        """Bin label of each value."""
        if self.kind == "numeric":
            values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
            idx = np.searchsorted(np.asarray(self.edges), values, side="left")
            idx = np.minimum(idx, len(self.edges) - 1)
            labels = np.asarray(self.labels[: len(self.edges)], dtype=object)[idx]
            return np.where(np.isnan(values), MISSING, labels)
        mapped = x.astype("object").where(x.notna(), MISSING).map(self.levels)
        return mapped.fillna(OTHER if OTHER in self.labels else MISSING).to_numpy()

    def transform(self, x: pd.Series) -> np.ndarray:
        """WoE of each value. A value in a bin unseen in training (e.g. missing) gets 0."""
        lookup = dict(zip(self.labels, self.woe, strict=True))
        return pd.Series(self.assign(x)).map(lookup).fillna(0.0).to_numpy(dtype=float)

    def table(self) -> pd.DataFrame:
        total_bad, total = sum(self.bads), sum(self.counts)
        return pd.DataFrame({
            "feature": self.name,
            "bin": self.labels,
            "loans": self.counts,
            "share": np.round(np.array(self.counts) / total, 4),
            "bad_rate": np.round(np.array(self.bads) / np.maximum(self.counts, 1), 4),
            "woe": np.round(self.woe, 4),
        }).assign(total_bad=total_bad)  # fmt: skip


def edge_label(lo: float, hi: float) -> str:
    def fmt(v: float) -> str:
        return f"{v:,.4g}" if abs(v) < 1e4 else f"{v:,.0f}"

    if np.isinf(lo):
        return f"<= {fmt(hi)}"
    if np.isinf(hi):
        return f"> {fmt(lo)}"
    return f"({fmt(lo)}, {fmt(hi)}]"


def fit_numeric(name: str, x: pd.Series, y: pd.Series) -> FeatureBins:
    values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    present = ~np.isnan(values)
    v, t = values[present], target[present]

    cuts = np.unique(np.quantile(v, np.linspace(0, 1, INITIAL_BINS + 1)[1:-1]))
    edges = [*cuts.tolist(), np.inf]
    idx = np.searchsorted(np.asarray(edges), v, side="left")
    counts = np.bincount(idx, minlength=len(edges)).astype(float)
    bads = np.bincount(idx, weights=t, minlength=len(edges))
    keep = counts > 0
    edges = [e for e, k in zip(edges, keep, strict=True) if k]
    edges[-1] = np.inf
    counts, bads = counts[keep], bads[keep]

    rates = bads / counts
    increasing = (
        bool(np.corrcoef(np.arange(len(rates)), rates)[0, 1] >= 0) if len(rates) > 1 else True
    )

    def regroup(new_edges: list[float]) -> tuple[list[float], list[float]]:
        i = np.searchsorted(np.asarray(new_edges), v, side="left")
        return (np.bincount(i, weights=t, minlength=len(new_edges)).tolist(),
                np.bincount(i, minlength=len(new_edges)).astype(float).tolist())  # fmt: skip

    edges = monotonic_merge(edges, bads.tolist(), counts.tolist(), increasing)
    b, n = regroup(edges)
    edges = merge_small(edges, b, n, MIN_BIN_SHARE * len(v))
    b, n = regroup(edges)

    lows = [-np.inf, *edges[:-1]]
    labels = [edge_label(lo, hi) for lo, hi in zip(lows, edges, strict=True)]
    bads_all, counts_all = list(b), list(n)
    if (~present).any():
        labels.append(MISSING)
        bads_all.append(float(target[~present].sum()))
        counts_all.append(float((~present).sum()))
    woe, iv = woe_iv(np.array(bads_all), np.array(counts_all) - np.array(bads_all))
    return FeatureBins(
        name, "numeric", labels, woe.tolist(), [int(x) for x in bads_all],
        [int(x) for x in counts_all], iv, edges=edges, increasing=increasing,
    )  # fmt: skip


def fit_categorical(name: str, x: pd.Series, y: pd.Series) -> FeatureBins:
    values = x.astype("object").where(x.notna(), MISSING)
    share = values.value_counts(normalize=True)
    levels = {lvl: (lvl if s >= MIN_LEVEL_SHARE else OTHER) for lvl, s in share.items()}
    labels_per_row = values.map(levels)
    grouped = pd.DataFrame({"label": labels_per_row, "y": y.to_numpy()}).groupby("label")["y"]
    stats = grouped.agg(["sum", "count"]).sort_values("sum", ascending=False)
    stats = stats.assign(rate=stats["sum"] / stats["count"]).sort_values("rate")
    woe, iv = woe_iv(stats["sum"].to_numpy(), (stats["count"] - stats["sum"]).to_numpy())
    labels = stats.index.astype(str).tolist()
    if OTHER not in labels:
        levels.setdefault(OTHER, OTHER)
    return FeatureBins(name, "categorical", labels, woe.tolist(),
                       stats["sum"].astype(int).tolist(), stats["count"].astype(int).tolist(),
                       iv, levels={str(k): v for k, v in levels.items()})  # fmt: skip


def fit(name: str, x: pd.Series, y: pd.Series) -> FeatureBins:
    if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
        return fit_numeric(name, x, y)
    return fit_categorical(name, x.astype("string"), y)
