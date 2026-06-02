# -*- coding: utf-8 -*-
"""Flask + tiny local LLM standardizer with incremental JSONL CLI output."""

from __future__ import annotations

import json
import os
import re
import sys
import difflib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request
from huggingface_hub import hf_hub_download
from llama_cpp import Llama  # CPU-only by default if N_GPU_LAYERS=0

app = Flask(__name__)

# ---------------- Model config ----------------
MODEL_REPO = os.getenv(
    "MODEL_REPO",
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
)
MODEL_FILE = os.getenv(
    "MODEL_FILE",
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
)

N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 2)))
N_CTX = int(os.getenv("N_CTX", "2048"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))  # 0 → CPU-only

CANON_UNIS_PATH = os.getenv("CANON_UNIS_PATH", "canon_universities.txt")
CANON_PROGS_PATH = os.getenv("CANON_PROGS_PATH", "canon_programs.txt")

# Precompiled, non-greedy JSON object matcher to tolerate chatter around JSON
JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)

# ---------------- Canonical lists + abbrev maps ----------------
def _read_lines(path: str) -> List[str]:
    """Read non-empty, stripped lines from a file (UTF-8)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        return []


CANON_UNIS = _read_lines(CANON_UNIS_PATH)
CANON_PROGS = _read_lines(CANON_PROGS_PATH)

ABBREV_UNI: Dict[str, str] = {
    r"(?i)^mcg(\.|ill)?$": "McGill University",
    r"(?i)^(ubc|u\.?b\.?c\.?)$": "University of British Columbia",
    r"(?i)^uoft$": "University of Toronto",
}

COMMON_UNI_FIXES: Dict[str, str] = {
    "McGiill University": "McGill University",
    "Mcgill University": "McGill University",
    # Normalize 'Of' → 'of'
    "University Of British Columbia": "University of British Columbia",
}

COMMON_PROG_FIXES: Dict[str, str] = {
    "Mathematic": "Mathematics",
    "Info Studies": "Information Studies",
}

# ---------------- Few-shot prompt ----------------
SYSTEM_PROMPT = (
    "You are a data cleaning assistant. Standardize degree program and university "
    "names.\n\n"
    "Rules:\n"
    "- Input provides a single string under key `program` that may contain both "
    "program and university.\n"
    "- Split into (program name, university name).\n"
    "- Trim extra spaces and commas.\n"
    '- Expand obvious abbreviations (e.g., "McG" -> "McGill University", '
    '"UBC" -> "University of British Columbia").\n'
    "- Use Title Case for program; use official capitalization for university "
    "names (e.g., \"University of X\").\n"
    '- Ensure correct spelling (e.g., "McGill", not "McGiill").\n'
    '- If university cannot be inferred, return "Unknown".\n\n'
    "Return JSON ONLY with keys:\n"
    "  standardized_program, standardized_university\n"
)

FEW_SHOTS: List[Tuple[Dict[str, str], Dict[str, str]]] = [
    (
        {"program": "Information Studies, McGill University"},
        {
            "standardized_program": "Information Studies",
            "standardized_university": "McGill University",
        },
    ),
    (
        {"program": "Information, McG"},
        {
            "standardized_program": "Information Studies",
            "standardized_university": "McGill University",
        },
    ),
    (
        {"program": "Mathematics, University Of British Columbia"},
        {
            "standardized_program": "Mathematics",
            "standardized_university": "University of British Columbia",
        },
    ),
]

_LLM: Llama | None = None
_LLM_LOCK = threading.Lock()   # llama.cpp is not thread-safe; serialize calls
_RESULT_CACHE: Dict[str, Dict[str, str]] = {}  # cache by input text to skip redundant calls


def _load_llm() -> Llama:
    """Download (or reuse) the GGUF file and initialize llama.cpp."""
    global _LLM
    if _LLM is not None:
        return _LLM

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir="models",
        local_dir_use_symlinks=False,
        force_filename=MODEL_FILE,
    )

    _LLM = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    return _LLM


def _split_fallback(text: str) -> Tuple[str, str]:
    """Simple, rules-first parser if the model returns non-JSON."""
    s = re.sub(r"\s+", " ", (text or "")).strip().strip(",")
    parts = [p.strip() for p in re.split(r",| at | @ ", s) if p.strip()]
    prog = parts[0] if parts else ""
    uni = parts[1] if len(parts) > 1 else ""

    # High-signal expansions
    if re.fullmatch(r"(?i)mcg(ill)?(\.)?", uni or ""):
        uni = "McGill University"
    if re.fullmatch(
        r"(?i)(ubc|u\.?b\.?c\.?|university of british columbia)",
        uni or "",
    ):
        uni = "University of British Columbia"

    # Title-case program; normalize 'Of' → 'of' for universities
    prog = prog.title()
    if uni:
        uni = re.sub(r"\bOf\b", "of", uni.title())
    else:
        uni = "Unknown"
    return prog, uni


def _best_match(name: str, candidates: List[str], cutoff: float = 0.86) -> str | None:
    """Fuzzy match via difflib (lightweight, Replit-friendly)."""
    if not name or not candidates:
        return None
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def _post_normalize_program(prog: str) -> str:
    """Apply common fixes, title case, then canonical/fuzzy mapping."""
    p = (prog or "").strip()
    p = COMMON_PROG_FIXES.get(p, p)
    p = p.title()
    if p in CANON_PROGS:
        return p
    match = _best_match(p, CANON_PROGS, cutoff=0.84)
    return match or p


def _post_normalize_university(uni: str) -> str:
    """Expand abbreviations, apply common fixes, capitalization, and canonical map."""
    u = (uni or "").strip()

    # Abbreviations
    for pat, full in ABBREV_UNI.items():
        if re.fullmatch(pat, u):
            u = full
            break

    # Common spelling fixes
    u = COMMON_UNI_FIXES.get(u, u)

    # Normalize 'Of' → 'of'
    if u:
        u = re.sub(r"\bOf\b", "of", u.title())

    # Canonical or fuzzy map
    if u in CANON_UNIS:
        return u
    match = _best_match(u, CANON_UNIS, cutoff=0.86)
    return match or u or "Unknown"


def _call_llm(program_text: str) -> Dict[str, str]:
    """Query the tiny LLM and return standardized fields (with cache + lock)."""
    if program_text in _RESULT_CACHE:
        return _RESULT_CACHE[program_text]

    llm = _load_llm()
    with _LLM_LOCK:  # serialize concurrent calls — llama.cpp is not thread-safe
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for x_in, x_out in FEW_SHOTS:
            messages.append(
                {"role": "user", "content": json.dumps(x_in, ensure_ascii=False)}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(x_out, ensure_ascii=False),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": json.dumps({"program": program_text}, ensure_ascii=False),
            }
        )

        out = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=128,
            top_p=1.0,
        )

        text = (out["choices"][0]["message"]["content"] or "").strip()
        try:
            match = JSON_OBJ_RE.search(text)
            obj = json.loads(match.group(0) if match else text)
            std_prog = str(obj.get("standardized_program", "")).strip()
            std_uni = str(obj.get("standardized_university", "")).strip()
        except Exception:
            std_prog, std_uni = _split_fallback(program_text)

        std_prog = _post_normalize_program(std_prog)
        std_uni = _post_normalize_university(std_uni)
        result = {
            "standardized_program": std_prog,
            "standardized_university": std_uni,
        }

    _RESULT_CACHE[program_text] = result
    return result


def _normalize_input(payload: Any) -> List[Dict[str, Any]]:
    """Accept either a list of rows or {'rows': [...]}."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    return []


@app.get("/")
def health() -> Any:
    """Simple liveness check."""
    return jsonify({"ok": True})


@app.post("/standardize")
def standardize() -> Any:
    """Standardize rows from an HTTP request and return JSON."""
    payload = request.get_json(force=True, silent=True)
    rows = _normalize_input(payload)

    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(_process_row(row))

    return jsonify({"rows": out})


def _build_program_text(row: Dict[str, Any]) -> str:
    """Combine program + university into one string for better LLM context.

    The LLM was designed for inputs like "Information Studies, McGill University".
    Giving it both fields produces more accurate standardization than program alone.
    """
    prog = (row or {}).get("program") or ""
    uni = (row or {}).get("university") or ""
    if prog and uni:
        return f"{prog}, {uni}"
    return prog or uni


def _process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize one row and attach llm-generated fields."""
    program_text = _build_program_text(row)
    result = _call_llm(program_text)
    row["llm-generated-program"] = result["standardized_program"]
    row["llm-generated-university"] = result["standardized_university"]
    return row


def _cli_process_file(
    in_path: str,
    out_path: str | None,
    workers: int,
    to_stdout: bool,
) -> None:
    """Process a JSON file using a thread pool and write a valid JSON array.

    Parallelisation strategy: ThreadPoolExecutor submits rows concurrently;
    the LLM calls are serialized internally by _LLM_LOCK, but threads overlap
    on the I/O and post-processing work. Cache hits bypass the lock entirely,
    making repeated program+university pairs nearly free.

    Args:
        in_path:   Path to input JSON file.
        out_path:  Path to write the output JSON array (ignored if to_stdout).
        workers:   Number of threads in the pool.
        to_stdout: If True, write to stdout instead of out_path.
    """
    with open(in_path, "r", encoding="utf-8") as f:
        rows = _normalize_input(json.load(f))

    total = len(rows)
    results: List[Dict[str, Any]] = [{}] * total  # pre-allocate to preserve order

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_row, row): i for i, row in enumerate(rows)}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            done += 1
            if done % 500 == 0 or done == total:
                cache_hits = total - len(_RESULT_CACHE)
                print(f"  {done}/{total} rows processed | cache size: {len(_RESULT_CACHE)}", file=sys.stderr)

    out_json = json.dumps(results, indent=2, ensure_ascii=False)

    if to_stdout:
        print(out_json)
    else:
        out_path = out_path or (in_path.replace(".json", "") + "_llm_extended.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"Saved {total} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standardize program/university with a tiny local LLM.",
    )
    parser.add_argument(
        "--file",
        help="Path to JSON input (list of rows or {'rows': [...]})",
        default=None,
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the HTTP server instead of CLI.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for the JSON array. Defaults to <input>_llm_extended.json.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write JSON array to stdout instead of a file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of parallel worker threads (default: half of CPU count).",
    )
    args = parser.parse_args()

    if args.serve or args.file is None:
        port = int(os.getenv("PORT", "8000"))
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        _cli_process_file(
            in_path=args.file,
            out_path=args.out,
            workers=args.workers,
            to_stdout=bool(args.stdout),
        )
