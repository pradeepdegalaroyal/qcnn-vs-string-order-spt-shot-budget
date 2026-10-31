"""Greyscale figures (200 dpi) and the numeric summary written to ``results/summary.json``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 200
METHODS = ["qcnn", "qcnn-no-pool", "string", "local-LR"]
LABELS = {
    "qcnn": "QCNN (trained pooling)", "qcnn-no-pool": "QCNN, pooling frozen",
    "string": "String-order threshold", "local-LR": "Logistic regression, 4 Pauli features",
}
GREY = {"qcnn": "0.15", "qcnn-no-pool": "0.5", "string": "0.75", "local-LR": "1.0"}
HATCH = {"qcnn": "", "qcnn-no-pool": "//", "string": "xx", "local-LR": ".."}
MARK = {"qcnn": "o", "qcnn-no-pool": "s", "string": "^", "local-LR": "D"}
LINE = {"qcnn": "-", "qcnn-no-pool": "--", "string": ":", "local-LR": "-."}
BIN_NAMES = {"near": "near (delta < 0.15)", "mid": "mid (0.15 to 0.4)", "far": "far (0.4 to 0.8)"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _per_seed(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)["acc"].agg(["mean", "std"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def fig_bars(acc: pd.DataFrame, path: Path, shots: int, n: int, n_train: int) -> None:
    """Grouped hatched bars: accuracy by distance bin for each method at a fixed shot budget."""
    d = acc[(acc.N == n) & (acc.n_train == n_train) & (acc.S == shots) & acc["bin"].isin(BIN_NAMES)]
    g = _per_seed(d, ["method", "bin"])
    methods = [m for m in METHODS if m in set(g.method)]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        s = g[g.method == m].set_index("bin").loc[list(BIN_NAMES)]
        ax.bar(np.arange(3) + (i - (len(methods) - 1) / 2) * w, s["mean"], w, yerr=s["std"],
               color=GREY[m], edgecolor="black", hatch=HATCH[m], capsize=3, label=LABELS[m])
    ax.set_xticks(range(3), [BIN_NAMES[b] for b in BIN_NAMES])
    ax.set_xlabel("distance |h1 - 1| of the test state from the critical field")
    ax.set_ylabel(f"classification accuracy at {shots} state copies")
    ax.set_ylim(0.5, 1.2)
    ax.set_title(f"Accuracy per distance bin, N = {n} qubits (mean and std over seeds)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    _save(fig, path)


def pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated by another one with fewer-or-equal cost and higher-or-equal quality."""
    keep = []
    for i, (c, q) in enumerate(points):
        dominated = any(
            (c2 <= c and q2 >= q) and (c2 < c or q2 > q) for j, (c2, q2) in enumerate(points) if j != i
        )
        if not dominated:
            keep.append(i)
    return keep


def fig_pareto(acc: pd.DataFrame, path: Path, sizes: list[int], n_train: int) -> None:
    """Cost (state copies) versus quality (accuracy near the critical point) with the Pareto front."""
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.6 * len(sizes), 4.3), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        d = acc[(acc.N == n) & (acc.n_train == n_train) & (acc["bin"] == "crit") & np.isfinite(acc.S)]
        g = _per_seed(d, ["method", "S"])
        pts, tags = [], []
        for m in METHODS:
            s = g[g.method == m].sort_values("S")
            if s.empty:
                continue
            ax.errorbar(s.S, s["mean"], yerr=s["std"], color="black", marker=MARK[m], ls=LINE[m],
                        mfc=GREY[m], capsize=2, label=LABELS[m])
            pts += list(zip(s.S, s["mean"]))
            tags += [m] * len(s)
        keep = sorted(pareto_front(pts), key=lambda i: pts[i][0])
        ax.plot([pts[i][0] for i in keep], [pts[i][1] for i in keep], ls="-", color="black", lw=0.8,
                marker="o", ms=13, mfc="none", mew=1.4, label="Pareto front")
        ax.set_xscale("log")
        ax.set_xlabel("state copies (shots) per test state")
        ax.set_ylabel("accuracy for |h1 - 1| < 0.4")
        ax.set_title(f"N = {n} qubits")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Shot cost versus accuracy near the phase transition")
    _save(fig, path)


def _annotated_heatmap(ax, mat: np.ndarray, xt: list[str], yt: list[str], fmt: str, vmin, vmax) -> None:
    im = ax.imshow(mat, cmap="Greys", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    ax.set_xticks(range(len(xt)), xt)
    ax.set_yticks(range(len(yt)), yt)
    mid = (vmin + vmax) / 2
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > mid else "black")
    plt.colorbar(im, ax=ax)


def fig_noise_heatmap(noise: pd.DataFrame, path: Path, n: int) -> None:
    """Accuracy gain (percentage points) of the QCNN over the string estimator over (gate error, readout error)."""
    d = noise[noise["bin"] == "crit"].copy()
    d["gain"] = 100 * (d.acc_qcnn - d.acc_string)
    piv = d.groupby(["p2", "r"])["gain"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    lim = max(1.0, float(np.abs(piv.values).max()))
    _annotated_heatmap(ax, piv.values, [f"{r:g}" for r in piv.columns], [f"{p:g}" for p in piv.index],
                       "+.1f", -lim, lim)
    ax.set_xlabel("readout flip probability per measured qubit")
    ax.set_ylabel("depolarising probability per two-qubit gate")
    ax.set_title(f"QCNN minus string-order accuracy (points), N = {n}" + chr(10) + "dark = QCNN better, light = string better", fontsize=10)
    _save(fig, path)


def fig_mps_heatmap(mps: pd.DataFrame, path: Path, n: int) -> None:
    """Mean absolute readout error of the QCNN on MPS-truncated inputs over (bond dimension, phase and distance)."""
    d = mps.copy()
    d["side"] = np.where(d.label > 0, "SPT", "PM")
    d["col"] = d.side + " " + d["bin"]
    cols = [f"{s} {b}" for s in ("SPT", "PM") for b in ("near", "mid", "far")]
    d["err"] = (d.out_chi - d.out_exact).abs()
    piv = d.groupby(["chi", "col"])["err"].mean().unstack()[cols]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _annotated_heatmap(ax, piv.values, cols, [str(c) for c in piv.index], ".3f", 0.0, max(1e-3, piv.values.max()))
    ax.set_xlabel("input phase (SPT or paramagnet, PM) and distance bin")
    ax.set_ylabel("MPS bond dimension chi")
    ax.set_title(f"Mean |QCNN output(chi) - QCNN output(exact)|, N = {n}")
    _save(fig, path)


def fig_ntrain(acc: pd.DataFrame, path: Path, n: int, shots: int) -> None:
    """Accuracy near the critical point versus the number of labelled training states."""
    d = acc[(acc.N == n) & (acc.S == shots) & (acc["bin"] == "crit")]
    g = _per_seed(d, ["method", "n_train"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in ("qcnn", "string", "local-LR"):
        s = g[g.method == m].sort_values("n_train")
        if len(s):
            ax.errorbar(s.n_train, s["mean"], yerr=s["std"], color="black", marker=MARK[m], ls=LINE[m],
                        mfc=GREY[m], capsize=3, label=LABELS[m])
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(g.n_train.unique()), [str(v) for v in sorted(g.n_train.unique())])
    ax.set_xlabel("labelled training states (half per phase)")
    ax.set_ylabel("accuracy for |h1 - 1| < 0.4")
    ax.set_title(f"Label scarcity at {shots} state copies, N = {n} qubits")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


def summarise(acc: pd.DataFrame, noise: pd.DataFrame, mps: pd.DataFrame, cost: pd.DataFrame, n_train: int,
              outs: pd.DataFrame | None = None) -> dict:
    """Headline numbers (mean and std over seeds) that the README quotes."""
    out: dict = {"shots_to_90pct": {}, "accuracy": {}, "noise": {}, "mps": {}, "compiled_cost": {}}
    d = acc[(acc.n_train == n_train)]
    for n in sorted(d.N.unique()):
        for m in sorted(d[d.N == n].method.unique()):
            sub = d[(d.N == n) & (d.method == m)]
            crit = _per_seed(sub[(sub["bin"] == "crit") & np.isfinite(sub.S)], ["S"])
            ok = crit[crit["mean"] >= 0.9]
            out["shots_to_90pct"][f"N{n}/{m}"] = int(ok.S.min()) if len(ok) else None
            for b in ("near", "mid", "far", "crit", "all"):
                for s in (100, 1000, np.inf):
                    one = sub[(sub["bin"] == b) & (sub.S == s)].acc
                    if len(one):
                        out["accuracy"][f"N{n}/{m}/{b}/S={s:g}"] = {
                            "mean": float(one.mean()), "std": float(one.std(ddof=1)) if len(one) > 1 else 0.0}
    for n_t in sorted(acc.n_train.unique()):
        for m in ("qcnn", "string", "local-LR"):
            one = acc[(acc.N == acc.N.min()) & (acc.n_train == n_t) & (acc.method == m) &
                      (acc["bin"] == "crit") & (acc.S == 300)].acc
            if len(one):
                out["accuracy"][f"ntrain={n_t}/{m}/crit/S=300"] = {
                    "mean": float(one.mean()), "std": float(one.std(ddof=1)) if len(one) > 1 else 0.0}
    if outs is not None:
        o = outs[(outs.n_train == n_train) & (outs.delta < 0.4)].copy()
        o["margin"] = (o.value - o.threshold).abs()
        out["margin_crit_region"] = {f"N{n}/{m}": float(v) for (n, m), v in o.groupby(["N", "method"])["margin"].mean().items()}
    if len(noise):
        crit = noise[noise["bin"] == "crit"].copy()
        crit["gain_pp"] = 100 * (crit.acc_qcnn - crit.acc_string)
        g = crit.groupby(["p2", "r"])["gain_pp"].agg(["mean", "std"]).reset_index()
        out["noise"]["gain_pp_by_p2_r"] = {f"p2={a:g},r={b:g}": {"mean": float(m), "std": float(0 if np.isnan(s) else s)}
                                           for a, b, m, s in g.itertuples(index=False)}
    if len(mps):
        m = mps.copy()
        m["err"] = (m.out_chi - m.out_exact).abs()
        for (chi, b), s in m.groupby(["chi", "bin"])["err"]:
            out["mps"][f"chi={chi}/{b}/mean_abs_output_error"] = float(s.mean())
        for chi, s in m.groupby("chi")["correct_chi"]:
            out["mps"][f"chi={chi}/accuracy_inf_shots"] = float(s.mean())
        out["mps"]["accuracy_inf_shots_exact_input"] = float(m.correct_exact.mean())
    if len(cost):
        for (n, conn), s in cost.groupby(["n_qubits", "connectivity"]):
            out["compiled_cost"][f"N{n}/{conn}"] = {
                "cx_mean": float(s.cx.mean()), "cx_std": float(s.cx.std(ddof=1)) if len(s) > 1 else 0.0,
                "depth_mean": float(s.depth.mean()), "logical_gates": int(s.logical_gates.iloc[0])}
    return out
