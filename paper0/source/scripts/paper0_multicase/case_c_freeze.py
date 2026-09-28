"""Case-C event freeze register: the machine-enforced precondition for real C.

Generation 20260826 (schema v1): authorised by
``data_paper0/extension_multicase_20260826/PLAN_AMENDMENT_CASE_C_20260830.md``
sections 2, 3 and 3.1 (site Dayekou, window unresolved pre-science).

Generation 20260905 (schema v2): authorised by
``data_paper0/extension_multicase_20260905/PLAN_MULTICASE_20260905.md``
(section 1, Case C = DEC_EBAO).  The author fixed the contemporaneous window
2015-12-01 ~ 2015-12-03 BY CONTRACT before any Case-C science; site = Ebao
with author-supplied namelist reference coordinates; the register's per-file
SHA-256 values remain the only attestable identity of which bytes the
declaration applies to.

Why this module exists: the ``.pt`` payloads carry no absolute time -- the top-level
object is a bare ``torch.Tensor`` with ``__metadata__ = None`` -- so a declared date
cannot be verified from the data.  Bytes are the only attestable identity, so Case C
may only become REAL against a register whose per-file SHA-256 values match the
payload exactly.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import mc_config as C

SCHEMA_V1 = "paper0_multicase/case_c_event_freeze/v1"
SCHEMA_V2 = "paper0_multicase/case_c_event_freeze/v2"
REGISTER_NAME = "CASE_C_EVENT_FREEZE.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_SELECTION_BASIS_V1 = (
    "wrf_numerical_completion_feasibility",
    "forcing_completeness",
)
ALLOWED_SELECTION_BASIS_V2 = (
    "author_contract_fixed_window",
)
UNRESOLVED = "UNRESOLVED_PRE_SCIENCE"

_SCHEMA = str(C.GEN.get("case_c_freeze_schema", "v1"))
if _SCHEMA == "v2":
    SCHEMA = SCHEMA_V2
    ALLOWED_SELECTION_BASIS = ALLOWED_SELECTION_BASIS_V2
else:
    SCHEMA = SCHEMA_V1
    ALLOWED_SELECTION_BASIS = ALLOWED_SELECTION_BASIS_V1


def register_path() -> Path:
    """Resolve lazily: callers redirect ``C.EXT`` via ``set_output_dirs`` after
    import, so binding the path at import time would read the wrong tree."""
    return C.EXT / REGISTER_NAME


def load() -> dict | None:
    """Return the register, or None when no date has been frozen yet."""
    path = register_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def selection_status() -> str:
    reg = load()
    if reg is None:
        return UNRESOLVED
    return str(reg.get("event_window", UNRESOLVED))


def _errors(reg: dict) -> list[str]:
    """Check the register's own integrity, independent of any payload."""
    errs: list[str] = []
    if reg.get("schema") != SCHEMA:
        errs.append(f"register schema {reg.get('schema')!r} != {SCHEMA!r}")
    if str(reg.get("case_id", "")).upper() != "C":
        errs.append("register case_id must be C")
    if _SCHEMA == "v2":
        expected_site = str(C.CASES["C"]["location"]).strip().lower()
        if expected_site not in str(reg.get("site", "")).lower():
            errs.append(f"register site must be {C.CASES['C']['location']}")
        if str(reg.get("replication_role", "")) != (
                "contemporaneous_cross_site_replication"):
            errs.append("register replication_role must be "
                        "contemporaneous_cross_site_replication")
        expected_window = str(C.CASES["C"].get("dates", "")).strip()
        if expected_window and str(reg.get("event_window", "")).strip() != expected_window:
            errs.append("register event_window must equal the author-fixed "
                        f"contemporaneous window {expected_window!r}")
    else:
        if "dayekou" not in str(reg.get("site", "")).lower():
            errs.append("register site must be Dayekou")
        if str(reg.get("replication_role", "")) != "independent_site_event":
            errs.append("register replication_role must be independent_site_event")
    if reg.get("science_metrics_consulted_for_selection") is not False:
        errs.append(
            "register must record science_metrics_consulted_for_selection = false; "
            "an outcome-aware date selection voids the pre-registration")
    basis = reg.get("selection_basis")
    if not isinstance(basis, list) or not basis:
        errs.append("register selection_basis must be a non-empty list")
    else:
        unknown = [b for b in basis if b not in ALLOWED_SELECTION_BASIS]
        if unknown:
            errs.append(
                f"selection_basis entries not permitted by the amendment: {unknown}; "
                f"allowed = {list(ALLOWED_SELECTION_BASIS)}")
    for key in ("event_window", "raw_subdir", "frozen_at_utc", "authorized_by"):
        value = str(reg.get(key) or "").strip()
        if not value:
            errs.append(f"register field {key} must be non-empty")
        elif key == "authorized_by" and value.lower() in {"null", "tbd", "agent", "auto"}:
            errs.append("register authorized_by must name the author, not a placeholder")
    hashes = reg.get("raw_sha256")
    if not isinstance(hashes, dict):
        errs.append("register raw_sha256 must be an object keyed by logical stem")
    else:
        missing = [s for s in C.RAW_FILE_STEMS if s not in hashes]
        extra = [s for s in hashes if s not in C.RAW_FILE_STEMS]
        if missing:
            errs.append(f"register raw_sha256 missing stems: {missing}")
        if extra:
            errs.append(f"register raw_sha256 has unknown stems: {extra}")
        bad = [s for s, h in hashes.items() if not HEX64.match(str(h))]
        if bad:
            errs.append(f"register raw_sha256 not 64-hex for: {sorted(bad)}")
    return errs


def verify_payload(files: dict[str, Path]) -> list[str]:
    """Return validation errors; empty list means the payload matches the freeze.

    ``files`` maps logical stem -> path, as resolved by ``validate_case_input``.
    Called on the real-install path only; a placeholder dry run never reaches it.
    """
    reg = load()
    if reg is None:
        return [
            f"Case C may not be installed as REAL without a dated event freeze "
            f"register ({REGISTER_NAME}); the amendment requires the event window "
            f"and the seven raw SHA-256 values to be frozen BEFORE any "
            f"subgrid-moment analysis is computed"]
    errs = _errors(reg)
    if errs:
        return errs
    frozen = {s: str(h).lower() for s, h in reg["raw_sha256"].items()}
    for stem in C.RAW_FILE_STEMS:
        path = files.get(stem)
        if path is None:
            errs.append(f"payload lacks {stem} named in the freeze register")
            continue
        observed = C.sha256_of(Path(path))
        if observed != frozen[stem]:
            errs.append(
                f"{stem}: payload sha256 {observed} != frozen {frozen[stem]} "
                f"(bytes are the date attestation; the payload is not the "
                f"window that was frozen at {reg.get('frozen_at_utc')})")
    return errs


def capture(raw_dir: Path, event_window: str, authorized_by: str,
            selection_basis: tuple[str, ...] = ALLOWED_SELECTION_BASIS,
            notes: str = "", target: Path | None = None) -> dict:
    """Freeze the current payload bytes.  Refuses to overwrite an existing register.

    Run this once, by the author, as soon as the real Case-C payload is in its
    registered location and BEFORE any science phase is executed.  ``target``
    lets isolated probes write into their own scratch tree; the canonical tree
    keeps the default resolution.
    """
    target = Path(target) if target is not None else register_path()
    if target.exists():
        raise FileExistsError(
            f"{target} already exists; the frozen window is immutable. "
            f"Supersede only by a new dated amendment, never by editing.")
    present = {p.name: p for p in raw_dir.glob("*.pt")}
    files: dict[str, Path] = {}
    for stem in C.RAW_FILE_STEMS:
        name = C.raw_filename("C", stem)
        if name not in present:
            raise FileNotFoundError(f"{name} (for logical stem {stem}) not in {raw_dir}")
        files[stem] = present[name]
    if _SCHEMA == "v2":
        payload = {
            "schema": SCHEMA,
            "case_id": "C",
            "site": C.CASES["C"]["location"],
            "replication_role": "contemporaneous_cross_site_replication",
            "event_window": event_window,
            "raw_subdir": raw_dir.name,
            "selection_basis": list(selection_basis),
            "science_metrics_consulted_for_selection": False,
            "authorized_by": authorized_by,
            "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "raw_sha256": {stem: C.sha256_of(files[stem]) for stem in C.RAW_FILE_STEMS},
            "amendment": "PLAN_MULTICASE_20260905.md",
            "coordinates_provenance": C.CASES["C"].get(
                "coordinates_provenance",
                "author-supplied namelist reference coordinates"),
            "notes": notes,
        }
    else:
        payload = {
            "schema": SCHEMA,
            "case_id": "C",
            "site": "Dayekou",
            "replication_role": "independent_site_event",
            "event_window": event_window,
            "raw_subdir": raw_dir.name,
            "selection_basis": list(selection_basis),
            "science_metrics_consulted_for_selection": False,
            "authorized_by": authorized_by,
            "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "raw_sha256": {stem: C.sha256_of(files[stem]) for stem in C.RAW_FILE_STEMS},
            "amendment": "PLAN_AMENDMENT_CASE_C_20260830.md",
            "notes": notes,
        }
    errs = _errors(payload)
    if errs:
        raise ValueError("refusing to write an invalid freeze register: "
                         + "; ".join(errs))
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--event-window")
    ap.add_argument("--authorized-by")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    if a.capture:
        out = capture(a.capture, a.event_window, a.authorized_by,
                      notes=a.notes)
        print(f"frozen -> {register_path()}\n{json.dumps(out, indent=2)}")
    else:
        reg = load()
        print(json.dumps({"register": REGISTER_NAME,
                          "path": str(register_path()),
                          "exists": reg is not None,
                          "status": selection_status(),
                          "errors": _errors(reg) if reg else []}, indent=2))
