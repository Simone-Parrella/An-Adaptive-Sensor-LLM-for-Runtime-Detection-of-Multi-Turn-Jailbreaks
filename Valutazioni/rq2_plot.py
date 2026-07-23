import json, os, statistics
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "."
CONFIGS  = {"A": "b", "B": "a", "C": "f", "D": "aa"}
MAL = {"granite": "ensemble_g.jsonl", "llama": "ensemble_l.jsonl", "mistral": "ensemble_m.jsonl"}
SENS = ["g", "l", "m"]
SNAME = {"g": "Granite", "l": "Llama", "m": "Mistral"}
OUTDIR = "figures"

SENSOR_COLORS  = {"g": "#4C72B0", "l": "#DD8452", "m": "#55A868"}
VERDICT_ORDER  = ["SAFE", "SUSPICIOUS", "RESISTED", "VIOLATION"] 
VERDICT_COLORS = {"SAFE": "#6FBF73", "SUSPICIOUS": "#E8C547",
                  "RESISTED": "#4C72B0", "VIOLATION": "#C44E52"}


def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def sensor_blocked(rec, s):
    return rec.get("final_sensor_states", {}).get(s, {}).get("blocked_at") is not None


def gather():
    block, vdist = {}, {}
    for cfg, folder in CONFIGS.items():
        path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(path):
            print(f"[config {cfg}] cartella '{folder}' assente, salto"); continue
        block[cfg] = {s: {} for s in SENS}
        vdist[cfg] = {s: Counter() for s in SENS}
        for tgt, fn in MAL.items():
            fpath = os.path.join(path, fn)
            if not os.path.exists(fpath):
                continue
            recs = load(fpath); n = len(recs)
            bcount = {s: 0 for s in SENS}
            for rec in recs:
                for s in SENS:
                    if sensor_blocked(rec, s): bcount[s] += 1
                for t in rec["turns"]:
                    for s in SENS:
                        v = t.get("sensors", {}).get(s, {}).get("verdict")
                        if v is None:          
                            continue
                        vdist[cfg][s][v] += 1
            for s in SENS:
                block[cfg][s][tgt] = bcount[s] / n
    return block, vdist


def fig1(block):
    cfgs = [c for c in CONFIGS if c in block]
    x = np.arange(len(cfgs)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, s in enumerate(SENS):
        means, lo, hi = [], [], []
        for c in cfgs:
            rates = list(block[c][s].values())
            m = statistics.mean(rates) if rates else 0
            means.append(m * 100)
            lo.append((m - min(rates)) * 100 if rates else 0)
            hi.append((max(rates) - m) * 100 if rates else 0)
        ax.bar(x + (i - 1) * w, means, w, label=SNAME[s], color=SENSOR_COLORS[s],
               yerr=[lo, hi], capsize=4, error_kw=dict(elinewidth=1.2, ecolor="#333"))
    ax.set_xticks(x); ax.set_xticklabels([f"Config {c}" for c in cfgs])
    ax.set_ylabel("Block rate (%)")
    ax.set_title("Block rate by sensor and configuration\n(error bars: spread across the three targets)")
    ax.legend(title="Sensor", frameon=False)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "fig1_sensor_vs_target.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("scritto", out)


def fig3(vdist):
    cfgs = [c for c in CONFIGS if c in vdist]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_w = 0.8; positions, labels, group_centers = [], [], []; pos = 0
    for s in SENS:
        start = pos
        for c in cfgs:
            tot = sum(vdist[c][s].values()) or 1   
            bottom = 0
            for v in VERDICT_ORDER:
                frac = vdist[c][s].get(v, 0) / tot * 100
                ax.bar(pos, frac, bar_w, bottom=bottom, color=VERDICT_COLORS[v],
                       label=v if (s == SENS[0] and c == cfgs[0]) else None)
                bottom += frac
            positions.append(pos); labels.append(c); pos += 1
        group_centers.append((start + pos - 1) / 2); pos += 0.8
    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8)
    for gc, s in zip(group_centers, SENS):
        ax.text(gc, -8, SNAME[s], ha="center", va="top", fontsize=11, fontweight="bold")
    ax.set_ylabel("Share of evaluated turns (%)"); ax.set_ylim(0, 100)
    ax.set_title("Verdict distribution per sensor across configurations")
    ax.legend(title="Verdict", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "fig3_verdict_stability.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("scritto", out)


def main():
    block, vdist = gather()
    if block: fig1(block)
    if vdist: fig3(vdist)


if __name__ == "__main__":
    main()
