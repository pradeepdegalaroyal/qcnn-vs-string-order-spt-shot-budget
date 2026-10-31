"""Reproduce every result and figure: ``python experiments/run_all.py`` (add ``--quick`` for a smoke run)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from spt_shotbench import plots  # noqa: E402
from spt_shotbench.studies import (  # noqa: E402
    Config, build_data, compiled_cost, main_study, mps_study, noise_grid,
)

torch.set_num_threads(6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke run (< 1 minute), writes to *_quick folders")
    args = ap.parse_args()
    cfg = Config.quick() if args.quick else Config()
    suffix = "_quick" if args.quick else ""
    res, fig = ROOT / f"results{suffix}", ROOT / f"figures{suffix}"
    res.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    log(f"ground states, N = {cfg.n_small} and {cfg.n_large}")
    small, large = build_data(cfg.n_small, cfg), build_data(cfg.n_large, cfg)

    log("study 1: accuracy versus state copies, small chain")
    acc, outs, models = main_study(cfg, small, cfg.seeds, cfg.n_train)
    for n_train in cfg.n_train_sweep:
        log(f"  label-scarcity sweep, n_train = {n_train}")
        r, o, _ = main_study(cfg, small, cfg.seeds, n_train, variants=("qcnn", "string", "local-LR"))
        acc, outs = acc + r, outs + o

    if cfg.n_large == cfg.n_small:  # quick mode: reuse the small chain as the "large" one
        models_large = {(s, "qcnn"): models[s, "qcnn"] for s in cfg.seeds_large}
    else:
        log("study 1b: large chain (QCNN transferred from the small chain, then fine-tuned)")
        init = {s: models[s, "qcnn"] for s in cfg.seeds_large}
        r, o, models_large = main_study(
            cfg, large, cfg.seeds_large, cfg.n_train, variants=("qcnn", "string"),
            steps=cfg.steps_large, init_from=init, dtype=torch.complex64,
        )
        acc, outs = acc + r, outs + o

    log("study 2: gate-noise x readout-noise grid")
    noise = noise_grid(cfg, small, models, cfg.seeds)

    log("study 3: matrix-product-state truncation of the input")
    mps = mps_study(cfg, large, models_large, cfg.seeds_large)

    acc_df, noise_df, mps_df = map(pd.DataFrame, (acc, noise, mps))
    acc_df.to_csv(res / "e1_accuracy.csv", index=False)
    pd.DataFrame(outs).to_csv(res / "e1_test_outputs.csv", index=False)
    log("study 4: compiled two-qubit-gate cost")
    cost_df = pd.DataFrame(compiled_cost(models) + compiled_cost(models_large))
    noise_df.to_csv(res / "e2_noise_grid.csv", index=False)
    mps_df.to_csv(res / "e3_mps_truncation.csv", index=False)
    cost_df.to_csv(res / "e4_compiled_cost.csv", index=False)
    params = {f"N{cfg.n_small}/{v}": int(m.n_params) for (s, v), m in models.items() if s == cfg.seeds[0]}
    params[f"N{cfg.n_large}/qcnn"] = int(models_large[cfg.seeds_large[0], "qcnn"].n_params)
    summary = plots.summarise(acc_df, noise_df, mps_df, cost_df, cfg.n_train, pd.DataFrame(outs))
    summary["parameters"] = params
    summary["train_loss"] = {
        f"N{m.n}/seed{s}/{v}": float(m.final_loss)
        for group in (models, models_large) for (s, v), m in group.items()
    }
    (res / "summary.json").write_text(json.dumps(summary, indent=2))

    log("figures")
    shots = 100 if 100 in cfg.shots else cfg.shots[0]
    plots.fig_bars(acc_df, fig / "fig1_accuracy_by_distance.png", shots, cfg.n_small, cfg.n_train)
    sizes = [cfg.n_small] if cfg.n_large == cfg.n_small else [cfg.n_small, cfg.n_large]
    plots.fig_pareto(acc_df, fig / "fig2_shots_pareto.png", sizes, cfg.n_train)
    plots.fig_noise_heatmap(noise_df, fig / "fig3_noise_heatmap.png", cfg.n_small)
    plots.fig_mps_heatmap(mps_df, fig / "fig4_mps_truncation.png", cfg.n_large)
    sweep_shots = 300 if 300 in cfg.shots else cfg.shots[-1]
    plots.fig_ntrain(acc_df, fig / "fig5_label_scarcity.png", cfg.n_small, sweep_shots)
    log("done")


if __name__ == "__main__":
    main()
