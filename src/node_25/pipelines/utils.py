from pathlib import Path
from typing import Tuple
import json
import csv
from typing import Tuple, Sequence

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _json_default(o):
    from pathlib import Path as _P
    import numpy as _np
    if isinstance(o, _P):
        return str(o)
    if isinstance(o, (_np.integer, _np.floating)):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)

def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))

def append_csv(row: dict, path: Path) -> None:
    ensure_dir(path.parent)
    write = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write:
            w.writeheader()
        w.writerow(row)

def parse_widths(s: str) -> Tuple[int, ...]:
    s = s.strip()
    return tuple(int(x) for x in s.split(",") if x) if s else ()

def widths_tag(ws: Sequence[int]) -> str:
    return "x".join(map(str, ws)) if ws else "none"

def num_to_token(x: float) -> str:
    # 0.1 -> 0p1, -0.05 -> m0p05
    s = f"{x:g}"
    return s.replace(".", "p").replace("-", "m")

def schedule_tag(fracs: Sequence[float], epochs: Sequence[int]) -> str:
    if not fracs:
        return "pt-none"
    f = "-".join(num_to_token(v) for v in fracs)
    e = "-".join(str(v) for v in epochs) if epochs else "auto"
    return f"pt_f{f}_e{e}"