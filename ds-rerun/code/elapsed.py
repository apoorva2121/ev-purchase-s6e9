"""Elapsed minutes since the Stage 0 clock start. Time is measured, never estimated."""
import datetime as _dt, pathlib as _p

_START_FILE = _p.Path(__file__).resolve().parent.parent / "artifacts" / "clock_start.txt"


def start_utc() -> _dt.datetime:
    txt = _START_FILE.read_text().strip().split("=", 1)[1]
    return _dt.datetime.strptime(txt, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)


def elapsed_min() -> float:
    now = _dt.datetime.now(_dt.timezone.utc)
    return round((now - start_utc()).total_seconds() / 60.0, 1)


if __name__ == "__main__":
    print(f"{elapsed_min()} min since {start_utc().isoformat()}")
