import json, os, csv, statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------- CONFIG -----------------------------
BASE_DIR = "."
CONFIGS  = {"A": "b", "B": "a", "C": "f", "D": "aa"}
MAL = {"granite": "ensemble_g.jsonl", "llama": "ensemble_l.jsonl", "mistral": "ensemble_m.jsonl"}
SENS = ["g", "l", "m"]
SNAME = {"g": "Granite", "l": "Llama", "m": "Mistral"}
MHJ_CSV = "mhj_dataset.csv"    
MIN_N   = 5                    
OUTDIR  = "figures"
# ------------------------------------------------------------------


def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def sensor_blocked(rec, s):
    return rec.get("final_sensor_states", {}).get(s, {}).get("blocked_at") is not None


def load_tactics():
    """row_index -> tactic, mappatura POSIZIONALE sull'ordine del CSV."""
    with open(MHJ_CSV) as f:
        rows = list(csv.DictReader(f))
    return {i: r["tactic"] for i, r in enumerate(rows)}


def gather(tactics):
    rate = {}
    counts = defaultdict(int)
    for cfg, folder in CONFIGS.items():
        path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(path):
            print(f"[config {cfg}] cartella '{folder}' assente, salto"); continue
        per_target = {s: defaultdict(list) for s in SENS}
        for tgt, fn in MAL.items():
            fpath = os.path.join(path, fn)
            if not os.path.exists(fpath):
                continue
            recs = load(fpath)
            blk = {s: defaultdict(int) for s in SENS}
            tot = defaultdict(int)
            for rec in recs:
                tac = tactics.get(rec["row_index"], "UNKNOWN")
                tot[tac] += 1
                counts[tac] += 1
                for s in SENS:
                    if sensor_blocked(rec, s):
                        blk[s][tac] += 1
            for s in SENS:
                for tac, n in tot.items():
                    per_target[s][tac].append(blk[s][tac] / n)
        rate[cfg] = {s: {tac: statistics.mean(v) for tac, v in per_target[s].items()}
                     for s in SENS}
    return rate, counts


def avg_over_configs(rate):
    """media sulle config -> avg[sensor][tactic]."""
    cfgs = list(rate.keys())
    tactics = sorted({t for c in cfgs for s in SENS for t in rate[c][s]})
    avg = {s: {} for s in SENS}
    for s in SENS:
        for t in tactics:
            vals = [rate[c][s][t] for c in cfgs if t in rate[c][s]]
            avg[s][t] = statistics.mean(vals) if vals else 0.0
    return avg, tactics


def main():
    tactics = load_tactics()
    rate, counts = gather(tactics)
    if not rate:
        print("nessuna config trovata"); return
    avg, tac_list = avg_over_configs(rate)
    n_cfg = len(rate)
    tac_list = sorted(tac_list, key=lambda t: -counts[t])

    print("Conteggio conversazioni per tattica (per dataset):")
    for t in tac_list:
        n = counts[t] // (n_cfg * 3)  
        flag = "  <-- sample piccolo" if n < MIN_N else ""
        print(f"  {t:28s} {n}{flag}")

    print("\nBLOCK-RATE per (tattica, sensore), media su config e target:")
    print(f'{"tattica":28s} {"Granite":>9s} {"Llama":>9s} {"Mistral":>9s}')
    for t in tac_list:
        print(f'{t:28s} ' + ' '.join(f'{avg[s][t]:9.1%}' for s in SENS))

    print("\nDettaglio per configurazione (block-rate, media sui target):")
    for cfg in rate:
        print(f' Config {cfg}:')
        for t in tac_list:
            print(f'   {t:28s} ' + ' '.join(f'{rate[cfg][s].get(t,0):7.1%}' for s in SENS))

    heat = np.array([[avg[s][t] * 100 for s in SENS] for t in tac_list])
    fig, ax = plt.subplots(figsize=(6.2, 0.7 * len(tac_list) + 1.5))
    im = ax.imshow(heat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(60, heat.max()))
    ax.set_xticks(range(len(SENS))); ax.set_xticklabels([SNAME[s] for s in SENS])
    ax.set_yticks(range(len(tac_list))); ax.set_yticklabels(tac_list)
    for i in range(len(tac_list)):
        for j in range(len(SENS)):
            val = heat[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    color="black" if val < 0.6 * heat.max() else "white", fontsize=9)
    ax.set_title("Average block rate by tactic and sensor\n(mean over targets and configurations)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Block rate (%)")
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "fig2_heatmap_tactic_sensor.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("\nscritto", out)


if __name__ == "__main__":
    main()