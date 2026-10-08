import requests
import sys
import os
import time
import json
import re

DEFAULT_HOST          = "http://localhost:11434"
DEFAULT_TARGET_MODEL  = "granite3.3:latest"
DEFAULT_SENSOR_MODEL  = "mistral-nemo:latest"
DEFAULT_CTX_THRESHOLD = 0.40
DEFAULT_CTX_RESERVE   = 0.20
FALLBACK_CTX          = 4096
CHARS_PER_TOKEN       = 4

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "PROMPT")
SSP = os.path.join(PROMPT_DIR, "ss_system_p.txt")
SMP = os.path.join(PROMPT_DIR, "sm_system_p.txt")

def get_model_context_length(model: str, host: str) -> int:
    try:
        r = requests.post(f"{host}/api/show", json={"name": model}, timeout=15)
        r.raise_for_status()
        data = r.json()
        for key, val in data.get("model_info", {}).items():
            kl = key.lower()
            if ("context_length" in kl or "ctx_length" in kl) and isinstance(val, int) and val > 0:
                return val
        params = data.get("parameters", "")
        if isinstance(params, str):
            for line in params.splitlines():
                if line.strip().lower().startswith("num_ctx"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[1])
                        except ValueError:
                            pass
    except Exception:
        pass
    return FALLBACK_CTX

def build_sensor_prompt(history: list, latest_response: str) -> str:
    lines = ["=== CONVERSATION HISTORY ==="]
    for m in history:
        role = m.get("role", "unknown").upper()
        body = m.get("body", m.get("content", ""))
        lines.append(f"[{role}]: {body}")
    lines.append("\n=== LATEST ASSISTANT RESPONSE TO EVALUATE ===")
    lines.append(latest_response)
    return "\n".join(lines)

def call_sensor(history, latest_response, model, host, max_retries=5):
    with open(SSP, "r") as f:
        ssp = f.read()

    user_content = build_sensor_prompt(history, latest_response)

    for attempt in range(1, max_retries + 1):
        raw = ollama_chat(
            messages=[{"role": "user", "content": user_content}],
            model=model,
            host=host,
            system=ssp,
            temperature=0.0,
            num_predict=128,
        )
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = raw[start:end]
                json_str = json_str.replace("“", '"').replace("”", '"')
                json_str = json_str.replace("‘", '"').replace("’", '"')
                return json.loads(json_str)

        except Exception:
            print(f"[WARN] Tentativo {attempt}/{max_retries} - Sensore non JSON: {raw[:120]}")
            time.sleep(0.5)

    return {"risk_delta": 1.0, "verdict": "SAFE", "reason": "Sensor output format invalid"}

def ollama_chat(messages: list, model: str, host: str,
                system: str = None, temperature: float = 0.7,
                num_predict: int = 2048) -> str:
    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)
    payload = {
        "model": model,
        "messages": payload_messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    try:
        r = requests.post(f"{host}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        print(f"\n[ERRORE] Ollama non raggiungibile su {host}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        return f"[ERRORE OLLAMA: {e}]"

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)

class ContextManager:
    """
    Classe che gestisce la finestra di contesto per simulare un sistema reale ma con un contesto limitato
    Struttura interna:
      summary_block = testo compresso delle conversazioni passate (None se vuoto)
      recent = messaggi recenti [{role, content}, ...]
    Trigger compressione: estimated_tokens > model_ctx * ctx_threshold * (1 - ctx_reserve)
    Alla compressione: i messaggi più vecchi di recent vengono riassunti
    e uniti al summary_block esistente. Vengono mantenuti gli ultimi
    KEEP_AFTER_SUMMARY messaggi recenti intatti.
    """

    KEEP_AFTER_SUMMARY = 3

    def __init__(self, model_ctx: int, ctx_threshold: float = DEFAULT_CTX_THRESHOLD,
                 ctx_reserve: float = DEFAULT_CTX_RESERVE,
                 summarizer_model: str = DEFAULT_SENSOR_MODEL, host: str = DEFAULT_HOST):
        self.model_ctx = model_ctx
        self.compress_at = int(model_ctx * ctx_threshold * (1.0 - ctx_reserve))
        self.summarizer_model = summarizer_model
        self.host = host
        self.summary_block = None
        self.recent = []
        self.n_compressions = 0

    def _estimated_tokens(self, extra_system: str = None) -> int:
        total = 0
        if extra_system:
            total += estimate_tokens(extra_system)
        if self.summary_block:
            total += estimate_tokens(self.summary_block) + 40
        for m in self.recent:
            total += estimate_tokens(m.get("content", "")) + 4
        return total

    def usage_ratio(self, extra_system: str = None) -> float:
        return self._estimated_tokens(extra_system) / self.model_ctx

    def build_ollama_messages(self) -> list:
        msgs = []
        if self.summary_block:
            header = (
                "[CONVERSATION MEMORY — summarized earlier turns]\n"
                "The following is a compressed summary of the conversation so far, "
                "including any risk signals observed:\n\n"
            )
            msgs.append({"role": "system", "content": header + self.summary_block})
        msgs.extend(self.recent)
        return msgs

    def add_message(self, role: str, content: str, extra_system: str = None) -> bool:
        self.recent.append({"role": role, "content": content})
        if self._estimated_tokens(extra_system) > self.compress_at:
            return self._compress()
        return False

    def _compress(self) -> bool:
        with open(SMP, "r") as f:
            smp = f.read()
        if len(self.recent) <= self.KEEP_AFTER_SUMMARY:
            return False
        to_compress = self.recent[:-self.KEEP_AFTER_SUMMARY]
        keep        = self.recent[-self.KEEP_AFTER_SUMMARY:]
        lines = []
        if self.summary_block:
            lines.append("[PREVIOUS SUMMARY]\n" + self.summary_block + "\n")
        lines.append("[NEW MESSAGES TO SUMMARIZE]")
        for m in to_compress:
            lines.append(f"[{m['role'].upper()}]: {m['content']}")
        summary = ollama_chat(
            messages=[{"role": "user", "content": "\n".join(lines)}],
            model=self.summarizer_model, host=self.host,
            system=smp,
            temperature=0.1, num_predict=512,
        )
        self.summary_block = summary.strip()
        self.recent = keep
        self.n_compressions += 1
        return True

    def inject_external_summary(self, summary: str):
        if self.summary_block:
            self.summary_block = summary + "\n\n[CONTINUATION FROM NEW SESSION]\n" + self.summary_block
        else:
            self.summary_block = summary

    def state_str(self, extra_system: str = None) -> str:
        ratio = self.usage_ratio(extra_system)
        w = 24
        filled = int(ratio * w)
        bar = "█" * filled + "░" * (w - filled)
        return (f"ctx [{bar}] ~{self._estimated_tokens(extra_system)}/{self.model_ctx} tok"
                f" ({ratio*100:.1f}%)  cmp={self.n_compressions}")