"""Small shared contracts: atomic files, fingerprints, RNG and exclusive locks."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import time

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
SCHEMA = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class ProgressReporter:
    """Publish throttled, atomic task progress for the parent tqdm display."""
    def __init__(self, output, interval=0.75):
        self.path = Path(output) / "progress.json"
        self.interval = interval
        self.last_write = 0.0
        self.last_phase = None

    def update(self, phase, current, total, unit="item", detail=None, force=False, **extra):
        phase_changed = phase != self.last_phase
        now_mono = time.monotonic()
        if not (force or phase_changed or current >= total or now_mono - self.last_write >= self.interval):
            return False
        payload = {"phase": phase, "current": int(current), "total": int(total),
                   "unit": unit, "updated": now(), **extra}
        if detail:
            payload["detail"] = str(detail)
        atomic_json(self.path, payload)
        self.last_write = now_mono
        self.last_phase = phase
        return True


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def code_files():
    paths = list((PACKAGE / "campaign").glob("*.py"))
    for folder in ["models", "configs", "data", "evaluation", "downstream/domain_adaptation"]:
        paths.extend((PACKAGE / folder).glob("*.py"))
    paths.extend(PACKAGE / "knowledge" / name for name in ["build.py", "schema.py"])
    paths.extend([REPO / "src/PECT_JEPA/temporal_1d/data/preprocessing.py",
                  REPO / "src/PECT_JEPA/spatiotemporal_5x5/data/preprocessing.py"])
    return sorted(set(paths))


def code_identity():
    return digest({str(p.relative_to(REPO)): sha256(p) for p in code_files()})


@contextlib.contextmanager
def exclusive_lock(path):
    """OS releases lock after crash. Never steal a live lock based on its age."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError(f"Another campaign owns lock: {path}") from None
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
