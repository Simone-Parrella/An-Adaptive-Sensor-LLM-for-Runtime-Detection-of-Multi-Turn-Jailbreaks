import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import copy

import pandas as pd
import requests

import model_req as mr
import risk_calculator as riskC

C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "cyan": "\033[36m", "magenta": "\033[35m", "white": "\033[37m",
    "blue": "\033[34m",
}

def cprint(text, color="reset", bold=False, end="\n"):
    pfx = C["bold"] if bold else ""
    print(f"{pfx}{C.get(color,'')}{text}{C['reset']}", end=end)

def vc_icon(v):
    return {"SAFE":"✅","RESISTED":"🛡️","SUSPICIOUS":"⚠️","VIOLATION":"🚨"}.get(v,"❓")
def rc(r):   return "green" if r < 3 else "yellow" if r < 6 else "red"
def tc(t):   return "green" if t > 7 else "yellow" if t > 5 else "red"



def extract_messages(row):
    messages, system = [], None
    for col in row.index:
        if not col.startswith("message_"):
            continue
        raw = row[col]
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            msg = json.loads(raw)
            role = msg.get("role", "").lower()
            body = msg.get("body", "")
            if not body:
                continue
            if role == "system":
                system = body
            else:
                messages.append({"role": role, "body": body})
        except (json.JSONDecodeError, AttributeError):
            continue
    return system, messages


def get_row_meta(row) -> dict:
    meta = {}
    for col in ["DERTA", "category", "Category"]:
        if col in row.index and pd.notna(row[col]):
            meta[col.lower()] = row[col]
    for col in row.index:
        if col not in ["DERTA"] and col not in meta and pd.notna(row.get(col)):
            val = row[col]
            if isinstance(val, (int, float)):
                meta[col] = val
    return meta


def call_active_sensors(
    history: list, latest_response: str,
    active_models: list[str], active_names: list[str], host: str,
    max_retries: int = 5
) -> list[dict]:
    results = []
    for name, model in zip(active_names, active_models):
        out = mr.call_sensor(
            history=history,
            latest_response=latest_response,
            model=model,
            host=host,
            max_retries=max_retries,
        )
        results.append({
            "name":       name,
            "model":      model,
            "risk_delta": float(out.get("risk_delta", 5.0)),
            "verdict":    out.get("verdict", "SUSPICIOUS"),
            "reason":     out.get("reason", ""),
        })
    return results


def run_conversation_ensemble(
    row_index: int, system_prompt: str, user_messages: list, meta: dict,
    target_model: str,
    sensor_models: list[str], sensor_names: list[str], sensor_weights: list[float],
    host: str, target_ctx: int,
    cooling: float, heat_scale: float,
    ctx_threshold: float, ctx_reserve: float,
    verbose: bool = True,
    inherited_risk: float = 0.0,
    inherited_threshold_state: dict = None,
    inherited_summary: str = None,
    session_turn_offset: int = 0,
) -> dict:
    
    sensors_state = {}
    for name, model in zip(sensor_names, sensor_models):
        re = riskC.RiskEngine(initial=inherited_risk, cooling=cooling, heat_scale=heat_scale)
        dt = riskC.DynamicThreshold(
            base=riskC.BASE_THRESHOLD,
            min_t=riskC.MIN_THRESHOLD,
            max_t=riskC.MAX_THRESHOLD,
            drop_scale=riskC.THRESHOLD_DROP_SCALE,
            early_boost=riskC.EARLY_TURNS_BOOST,
            early_decay=riskC.EARLY_DECAY,
            recovery=riskC.THRESHOLD_RECOVERY,
        )
        if inherited_threshold_state:
            dt.load_state(copy.deepcopy(inherited_threshold_state))
            
        sensors_state[name] = {
            "model": model,
            "risk_engine": re,
            "dyn_thresh": dt,
            "blocked_at_turn": None,
            "final_risk": None,
            "final_threshold": None
        }

    ctx_mgr = mr.ContextManager(
        model_ctx=target_ctx,
        ctx_threshold=ctx_threshold,
        ctx_reserve=ctx_reserve,
        summarizer_model=sensor_models[0],
        host=host,
    )
    if inherited_summary:
        ctx_mgr.inject_external_summary(inherited_summary)

    sensor_history = []

    result = {
        "row_index":        row_index,
        "meta":             meta,
        "system_prompt":    system_prompt,
        "target_model":     target_model,
        "sensor_models":    sensor_models,
        "sensor_names":     sensor_names,
        "cooling":          cooling,
        "heat_scale":       heat_scale,
        "target_ctx_tokens": target_ctx,
        "turns":            [],
        "all_blocked_at":   None,
        "completed":        False,
        "n_compressions":   0,
        "ts":               datetime.now().isoformat(),
    }

    if verbose:
        print("\n" + "═" * 72)
        cprint(f"  SESSION #{row_index}  target={target_model}", "cyan", bold=True)
        cprint(f"  Independent sensors: {' | '.join(sensor_names)}", "dim")
        print("═" * 72)

    for local_turn, user_msg in enumerate(user_messages, 1):
        global_turn = session_turn_offset + local_turn
        user_body = user_msg.get("body", user_msg.get("content", ""))

        if verbose:
            print()
            cprint(f"[Turn {local_turn}/{len(user_messages)}  global={global_turn}]",
                   "magenta", bold=True)
            cprint("👤 USER: ", "cyan", bold=True, end="")
            print(user_body[:200] + ("..." if len(user_body) > 200 else ""))

        compressed = ctx_mgr.add_message("user", user_body, extra_system=system_prompt)
        if compressed and verbose:
            cprint("  📦 Context COMPRESSION!", "blue", bold=True)

        sensor_history.append({"role": "user", "body": user_body})

        t0 = time.time()
        response = mr.ollama_chat(
            messages=ctx_mgr.build_ollama_messages(),
            model=target_model, host=host,
            system=system_prompt, temperature=0.7,
        )
        elapsed_target = time.time() - t0

        if verbose:
            cprint(f"🤖 TARGET ({elapsed_target:.1f}s): ", "yellow", bold=True, end="")
            print(response[:280] + ("..." if len(response) > 280 else ""))

        active_names = [n for n, s in sensors_state.items() if s["blocked_at_turn"] is None]
        active_models = [sensors_state[n]["model"] for n in active_names]

        t0 = time.time()
        sensor_outputs = []
        if active_names:
            sensor_outputs = call_active_sensors(
                history=sensor_history,
                latest_response=response,
                active_models=active_models,
                active_names=active_names,
                host=host,
            )
        elapsed_sensors = time.time() - t0

        if verbose:
            print(f"\n  📊 SENSOR EVALUATION ({elapsed_sensors:.1f}s):")

        turn_sensors_log = {}
        for name in sensor_names:
            state = sensors_state[name]
            if state["blocked_at_turn"] is not None:
                turn_sensors_log[name] = {
                    "model": state["model"],
                    "status": "ALREADY_BLOCKED",
                    "blocked_at": state["blocked_at_turn"]
                }
                if verbose:
                    cprint(f"     {name:10s}: ⛔ [ALREADY BLOCKED at turn {state['blocked_at_turn']}]", "dim")
                continue

            out = next(s for s in sensor_outputs if s["name"] == name)
            delta = out["risk_delta"]
            verdict = out["verdict"]

            new_thresh = state["dyn_thresh"].update(delta, verdict, global_turn)
            new_risk = state["risk_engine"].update(delta, verdict)
            is_blocked = state["risk_engine"].is_blocked(new_thresh)

            turn_sensors_log[name] = {
                "model": state["model"],
                "status": "BLOCKED_NOW" if is_blocked else "ACTIVE",
                "risk_delta": delta,
                "verdict": verdict,
                "reason": out["reason"],
                "risk_after": round(new_risk, 3),
                "threshold_after": round(new_thresh, 3),
            }

            if verbose:
                icon = vc_icon(verdict)
                color = "red" if is_blocked else "white"
                cprint(f"     {name:10s}: {icon} [{verdict:10s}]  Δ={delta:.2f}  {out['reason'][:50]}...", color)
                cprint(f"                Risk: {state['risk_engine'].risk_bar()}  Threshold: {state['dyn_thresh'].threshold_bar()}", "dim")

            if is_blocked:
                state["blocked_at_turn"] = local_turn
                state["final_risk"] = round(new_risk, 3)
                state["final_threshold"] = round(new_thresh, 3)

        all_blocked = all(s["blocked_at_turn"] is not None for s in sensors_state.values())

        result["turns"].append({
            "turn":         local_turn,
            "global_turn":  global_turn,
            "user":         user_body,
            "response":     response if not all_blocked else "[PERMANENTLY BLOCKED]",
            "sensors":      turn_sensors_log,
            "all_blocked":  all_blocked,
            "ctx_tokens_est": ctx_mgr._estimated_tokens(system_prompt),
            "compressed_this_turn": compressed,
            "elapsed_target_s":     round(elapsed_target, 3),
            "elapsed_sensors_s":    round(elapsed_sensors, 3),
        })

        if all_blocked:
            if verbose:
                cprint("\n  ⛔ CHAT BLOCKED — all sensors exceeded their thresholds!", "red", bold=True)
            
            final_states = {
                n: {
                    "blocked_at": s["blocked_at_turn"], 
                    "final_risk": s["final_risk"], 
                    "final_threshold": s["final_threshold"]
                } for n, s in sensors_state.items()
            }
            
            result.update({
                "all_blocked_at": local_turn,
                "final_sensor_states": final_states,
                "n_compressions":  ctx_mgr.n_compressions,
            })
            return result

        compressed2 = ctx_mgr.add_message("assistant", response, extra_system=system_prompt)
        if compressed2 and verbose:
            cprint("  📦 Context COMPRESSION (after response)!", "blue", bold=True)
        sensor_history.append({"role": "assistant", "body": response})

    final_states = {}
    for n, s in sensors_state.items():
        fr = s["final_risk"] if s["final_risk"] is not None else round(s["risk_engine"].risk, 3)
        ft = s["final_threshold"] if s["final_threshold"] is not None else round(s["dyn_thresh"].value, 3)
        final_states[n] = {
            "blocked_at": s["blocked_at_turn"],
            "final_risk": fr,
            "final_threshold": ft
        }

    result.update({
        "completed":       True,
        "final_sensor_states": final_states,
        "n_compressions":  ctx_mgr.n_compressions,
    })

    if verbose:
        cprint("\n  ✅ Session completed (not all sensors blocked).", "green", bold=True)

    return result


def run_ensemble_mode(
    sessions: list,
    target_models: list[str], target_names: list[str],
    sensor_models: list[str], sensor_names: list[str], sensor_weights: list[float],
    host: str, cooling: float, heat_scale: float,
    ctx_threshold: float, ctx_reserve: float,
    output_dir: Path, verbose: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cprint(f"\n{'═'*72}", "magenta")
    cprint(f"  ENSEMBLE MODE — {len(target_models)} target  ×  {len(sensor_models)} independent sensors",
           "magenta", bold=True)
    cprint(f"{'═'*72}\n", "magenta")

    for t_name, t_model in zip(target_names, target_models):
        out_path = output_dir / f"ensemble_{t_name}.jsonl"
        cprint(f"\n{'▓'*72}", "cyan")
        cprint(f"  TARGET: {t_model}  →  {out_path}", "cyan", bold=True)
        cprint(f"{'▓'*72}", "cyan")

        target_ctx = mr.get_model_context_length(t_model, host)

        with open(out_path, "w", encoding="utf-8") as f:
            for row_idx, sys_prompt, user_msgs, meta in sessions:
                result = run_conversation_ensemble(
                    row_index=row_idx,
                    system_prompt=sys_prompt,
                    user_messages=user_msgs,
                    meta=meta,
                    target_model=t_model,
                    sensor_models=sensor_models,
                    sensor_names=sensor_names,
                    sensor_weights=sensor_weights,
                    host=host,
                    target_ctx=target_ctx,
                    cooling=cooling,
                    heat_scale=heat_scale,
                    ctx_threshold=ctx_threshold,
                    ctx_reserve=ctx_reserve,
                    verbose=verbose,
                )
                line = json.dumps(result, ensure_ascii=False)
                f.write(line + "\n")
                f.flush()
                if not verbose:
                    print(line)

        cprint(f"  ✅ {out_path} written.", "green", bold=True)

    cprint(f"\n{'═'*72}", "green")
    cprint(f"  ENSEMBLE completed — files in: {output_dir}", "green", bold=True)
    cprint(f"{'═'*72}\n", "green")



def main():
    parser = argparse.ArgumentParser(
        description="Ensemble sensor — 1 target answers, N sensors evaluate in parallel until ALL of them block",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local test with small models (3 rows)
  python sensor_test.py \\
      --dataset mhj_dataset.csv --rows 0-2 \\
      --target-models qwen2.5:1.5b,tinyllama:latest,phi3:mini \\
      --sensor-models qwen2.5:1.5b,tinyllama:latest,phi3:mini \\
      --output-dir local_test/ --quiet
"""
    )

    parser.add_argument("--dataset", "-d", required=True, help="Path to the CSV dataset")
    parser.add_argument("--row",  type=int, default=None, help="Single row")
    parser.add_argument("--rows", help="E.g. 0,1,2 or 0-9")
    parser.add_argument("--all",  action="store_true", help="All rows")
    parser.add_argument("--target-models", required=True,
                        help="Comma-separated target models")
    parser.add_argument("--sensor-models", required=True,
                        help="Comma-separated sensor models")
    parser.add_argument("--sensor-weights", default=None,
                        help="Sensor weights (no effect with independent blocking, kept for backward compatibility)")
    parser.add_argument("--target-names", default=None,
                        help="Labels for the targets (default: g,l,m,...)")
    parser.add_argument("--sensor-names", default=None,
                        help="Labels for the sensors (default: g,l,m,...)")

    parser.add_argument("--host", default=mr.DEFAULT_HOST)
    parser.add_argument("--output-dir", "-o", default="ensemble_results/",
                        help="Output folder (default: ensemble_results/)")

    parser.add_argument("--cooling",        type=float, default=riskC.DEFAULT_COOLING)
    parser.add_argument("--heat-scale",     type=float, default=riskC.DEFAULT_HEAT_SCALE)
    parser.add_argument("--base-threshold", type=float, default=riskC.BASE_THRESHOLD)
    parser.add_argument("--min-threshold",  type=float, default=riskC.MIN_THRESHOLD)
    parser.add_argument("--drop-scale",     type=float, default=riskC.THRESHOLD_DROP_SCALE)
    parser.add_argument("--early-boost",    type=float, default=riskC.EARLY_TURNS_BOOST)
    parser.add_argument("--recovery",       type=float, default=riskC.THRESHOLD_RECOVERY)
    parser.add_argument("--max-threshold",  type=float, default=riskC.MAX_THRESHOLD)
    parser.add_argument("--early-decay",    type=float, default=riskC.EARLY_DECAY)
    parser.add_argument("--ctx-threshold",  type=float, default=mr.DEFAULT_CTX_THRESHOLD)
    parser.add_argument("--ctx-reserve",    type=float, default=mr.DEFAULT_CTX_RESERVE)
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()
    riskC.BASE_THRESHOLD       = args.base_threshold
    riskC.MIN_THRESHOLD        = args.min_threshold
    riskC.THRESHOLD_DROP_SCALE = args.drop_scale
    riskC.EARLY_TURNS_BOOST    = args.early_boost
    riskC.THRESHOLD_RECOVERY   = args.recovery
    riskC.MAX_THRESHOLD        = args.max_threshold
    riskC.EARLY_DECAY          = args.early_decay
    verbose = not args.quiet

    try:
        r = requests.get(f"{args.host}/api/tags", timeout=5)
        avail = [m["name"] for m in r.json().get("models", [])]
        if verbose:
            cprint(f"Ollama  : {args.host}", "cyan")
            cprint(f"Models  : {', '.join(avail)}", "dim")
    except Exception:
        cprint(f"[ERROR] Ollama not reachable at {args.host}", "red")
        sys.exit(1)

    target_models = [m.strip() for m in args.target_models.split(",") if m.strip()]
    sensor_models = [m.strip() for m in args.sensor_models.split(",") if m.strip()]

    default_labels = ["g", "l", "m", "d", "e", "f"]
    target_names = (
        [n.strip() for n in args.target_names.split(",") if n.strip()]
        if args.target_names
        else default_labels[:len(target_models)]
    )
    sensor_names = (
        [n.strip() for n in args.sensor_names.split(",") if n.strip()]
        if args.sensor_names
        else default_labels[:len(sensor_models)]
    )

    if args.sensor_weights:
        sensor_weights = [float(w.strip()) for w in args.sensor_weights.split(",")]
        if len(sensor_weights) != len(sensor_models):
            cprint("[ERROR] The number of weights must match the number of sensors", "red")
            sys.exit(1)
    else:
        sensor_weights = [1.0] * len(sensor_models)

    if verbose:
        cprint(f"Target  : {list(zip(target_names, target_models))}", "cyan")
        cprint(f"Sensors : {list(zip(sensor_names, sensor_models))}", "cyan")
        cprint(f"Config  : cooling={args.cooling}  heat_scale={args.heat_scale}  base_thr={args.base_threshold}", "dim")
    try:
        df = pd.read_csv(args.dataset)
    except FileNotFoundError:
        cprint(f"[ERROR] File not found: {args.dataset}", "red")
        sys.exit(1)

    if verbose:
        cprint(f"Dataset : {args.dataset} ({len(df)} rows)", "cyan")

    if args.all:
        row_indices = list(range(len(df)))
    elif args.rows:
        row_indices = []
        for part in args.rows.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                row_indices.extend(range(int(a), int(b) + 1))
            else:
                row_indices.append(int(part))
    elif args.row is not None:
        row_indices = [args.row]
    else:
        row_indices = [0]
        if verbose:
            cprint("No row specified, using row 0", "yellow")

    sessions = []
    for row_idx in row_indices:
        if row_idx >= len(df):
            continue
        row = df.iloc[row_idx]
        sys_prompt, user_msgs = extract_messages(row)
        meta = get_row_meta(row)
        if not user_msgs:
            if verbose:
                cprint(f"[SKIP] Row {row_idx}: no messages", "yellow")
            continue
        sessions.append((row_idx, sys_prompt, user_msgs, meta))

    if not sessions:
        cprint("[ERROR] No valid session.", "red")
        sys.exit(1)

    cprint(f"Sessions: {len(sessions)}", "cyan")

    run_ensemble_mode(
        sessions=sessions,
        target_models=target_models,
        target_names=target_names,
        sensor_models=sensor_models,
        sensor_names=sensor_names,
        sensor_weights=sensor_weights,
        host=args.host,
        cooling=args.cooling,
        heat_scale=args.heat_scale,
        ctx_threshold=args.ctx_threshold,
        ctx_reserve=args.ctx_reserve,
        output_dir=Path(args.output_dir),
        verbose=verbose,
    )

if __name__ == "__main__":
    main()