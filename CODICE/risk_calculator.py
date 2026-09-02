import math

BASE_THRESHOLD        = 8.0
MIN_THRESHOLD         = 7.0
MAX_THRESHOLD         = 10.0
THRESHOLD_DROP_SCALE  = 1.0
EARLY_TURNS_BOOST     = 2.0
EARLY_DECAY           = 0.6
THRESHOLD_RECOVERY    = 0.2

DEFAULT_COOLING       = 0.75
DEFAULT_HEAT_SCALE    = 1.5
INITIAL_RISK          = 0.0

class DynamicThreshold:
    def __init__(self, base=BASE_THRESHOLD, min_t=MIN_THRESHOLD, max_t=MAX_THRESHOLD,
                 drop_scale=THRESHOLD_DROP_SCALE, early_boost=EARLY_TURNS_BOOST,
                 early_decay=EARLY_DECAY, recovery=THRESHOLD_RECOVERY):
        self.threshold = base
        self.base = base
        self.min_t = min_t
        self.max_t = max_t
        self.drop_scale = drop_scale
        self.early_boost = early_boost
        self.early_decay = early_decay
        self.recovery_base = recovery
        self._n_safe = 0

    def _early_factor(self, turn: int) -> float:
        return 1.0 + (self.early_boost - 1.0) * (self.early_decay ** (turn - 1))

    def update(self, risk_delta: float, verdict: str, turn: int) -> float:
        if verdict == "SAFE":
            self._n_safe += 1
            recovery = self.recovery_base / math.log2(2 + self._n_safe)
            self.threshold = min(self.max_t, self.threshold + recovery)
        else:
            self._n_safe = 0
            ef = self._early_factor(turn)
            self.threshold = max(self.min_t, self.threshold - self.drop_scale * risk_delta * ef)
        return self.threshold

    @property
    def value(self) -> float:
        return self.threshold

    def threshold_bar(self, w=24) -> str:
        filled = int((self.threshold / 10.0) * w)
        return f"[{'█'*filled + '░'*(w-filled)}] {self.threshold:.2f}/10"

    def state_dict(self) -> dict:
        return {"threshold": round(self.threshold, 3), "n_safe": self._n_safe}

    def load_state(self, state: dict):
        self.threshold = state.get("threshold", self.base)
        self._n_safe   = state.get("n_safe", 0)


class RiskEngine:
    def __init__(self, initial=INITIAL_RISK, cooling=DEFAULT_COOLING, heat_scale=DEFAULT_HEAT_SCALE):
        self.risk = initial
        self.cooling = cooling
        self.heat_scale = heat_scale

    def update(self, risk_delta: float, verdict: str) -> float:
        if verdict == "SAFE":
            self.risk *= self.cooling
        else:
            self.risk += risk_delta * self.heat_scale
        self.risk = max(0.0, min(10.0, self.risk))
        return self.risk

    def is_blocked(self, threshold: float) -> bool:
        return self.risk >= threshold
    
    def risk_bar(self, w=24) -> str:
        filled = int((self.risk / 10.0) * w)
        return f"[{'█'*filled + '░'*(w-filled)}] {self.risk:.2f}/10"