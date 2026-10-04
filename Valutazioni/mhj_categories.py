import argparse
import json
import os
import urllib.request
import pandas as pd

HARMBENCH_URL = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
                 "data/behavior_datasets/harmbench_behaviors_text_all.csv")
QID_OFFSET = 2 

CATEGORY_NAMES = {
    "misinformation_disinformation": "Misinformation & disinformation",
    "illegal": "Illegal activities",
    "harassment_bullying": "Harassment & bullying",
    "harmful": "Harmful content",
    "cybercrime_intrusion": "Cybercrime & intrusion",
    "chemical_biological": "Chemical & biological",
    "copyright": "Copyright",
}


def load_harmbench(path):
    if path is None:
        path = "harmbench_behaviors_text_all.csv"
        if not os.path.exists(path):
            print(f"Downloading HarmBench behaviours from {HARMBENCH_URL}")
            urllib.request.urlretrieve(HARMBENCH_URL, path)
    return pd.read_csv(path, encoding="utf-8")


def count_user_turns(row):
    n = 0
    for col in row.index:
        if not col.startswith("message_"):
            continue
        val = row[col]
        if not isinstance(val, str):
            continue
        try:
            role = json.loads(val).get("role")
        except (json.JSONDecodeError, AttributeError):
            role = "user"
        if role == "user":
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mhj", default="mhj_dataset.csv")
    parser.add_argument("--harmbench", default=None)
    parser.add_argument("--out", default="mhj_with_categories.csv")
    args = parser.parse_args()

    mhj = pd.read_csv(args.mhj, encoding="utf-8")
    hb = load_harmbench(args.harmbench)

    idx = mhj["question_id"] - QID_OFFSET
    if idx.min() < 0 or idx.max() >= len(hb):
        raise ValueError("question_id out of range for the HarmBench file provided")

    mhj["behavior"] = hb["Behavior"].values[idx]
    mhj["behavior_id"] = hb["BehaviorID"].values[idx]
    mhj["category"] = hb["SemanticCategory"].values[idx]
    mhj["user_turns"] = mhj.apply(count_user_turns, axis=1)

    print("\nSanity check (submission message -> HarmBench behaviour):")
    for _, r in mhj.drop_duplicates("question_id").head(5).iterrows():
        print(f"  [{r.question_id}] {str(r.submission_message)[:70]}")
        print(f"        -> {r.behavior[:70]}  ({r.category})")

    n = len(mhj)
    summary = (mhj.groupby("category")
               .agg(conversations=("question_id", "size"),
                    behaviours=("question_id", "nunique"))
               .sort_values("conversations", ascending=False))
    summary["share"] = summary["conversations"] / n * 100

    print(f"\nConversations: {n}   user turns: {mhj.user_turns.sum()}   "
          f"distinct behaviours: {mhj.question_id.nunique()}")
    print(f"User turns per conversation: mean {mhj.user_turns.mean():.1f}, "
          f"median {mhj.user_turns.median():.0f}, max {mhj.user_turns.max()}, "
          f"<=3 turns {(mhj.user_turns <= 3).mean():.1%}\n")

    print(f'{"Category":34s} {"Conv.":>6s} {"Share":>7s} {"Behav.":>7s}')
    for cat, r in summary.iterrows():
        print(f'{CATEGORY_NAMES.get(cat, cat):34s} {int(r.conversations):6d} '
              f'{r.share:6.1f}% {int(r.behaviours):7d}')

    print("\nCategory x tactic:")
    print(pd.crosstab(mhj["category"], mhj["tactic"]).to_string())

    print("\n% --- LaTeX rows ---")
    for cat, r in summary.iterrows():
        name = CATEGORY_NAMES.get(cat, cat).replace("&", "\\&")
        print(f"{name} & {int(r.conversations)} & {r.share:.1f}\\% & {int(r.behaviours)} \\\\\n\\hline")
    print(f"Total & {n} & 100\\% & {mhj.question_id.nunique()} \\\\\n\\hline")

    cols = ["question_id", "behavior_id", "behavior", "category", "tactic",
            "Source", "user_turns"]
    mhj[cols].to_csv(args.out, index=False, encoding="utf-8")
    print(f"\nWritten {args.out}")


if __name__ == "__main__":
    main()