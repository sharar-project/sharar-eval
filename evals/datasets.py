"""Pinned datasets: every file an eval reads lives under $SHARAR_RESULTS/datasets (default: sharar's,
$XDG_DATA_HOME/sharar or ~/.local/share/sharar), is fetched from a pinned source and checked against
evals/datasets.sha256.

    sharar-eval datasets fetch     # download what is missing, then check everything
    sharar-eval datasets check     # check only
"""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path



def results_root() -> Path:
    """$SHARAR_RESULTS, else sharar's default results directory."""
    if root := os.environ.get("SHARAR_RESULTS"):
        return Path(root).expanduser()
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "sharar"


ROOT = results_root() / "datasets"
PINS = Path(__file__).resolve().parent / "datasets.sha256"

GH = "https://raw.githubusercontent.com"
URL_NLP = f"{GH}/google-research/url-nlp/452a21ad3dae5668c06ceeac21ff073e1e40f9be/mgsm"
NLTK = f"{GH}/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers"
MMLU = "https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/all"
HOTPOT = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor"
FILES = {  # dataset path: pinned source URL
    "accuracy/gsm8k_test.jsonl": f"{GH}/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
    "accuracy/HumanEval.jsonl.gz": f"{GH}/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz",
    "accuracy/mbpp.jsonl": f"{GH}/google-research/google-research/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/mbpp.jsonl",
    **{f"accuracy/mgsm/mgsm_{lang}.tsv": f"{URL_NLP}/mgsm_{lang}.tsv"
       for lang in ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")},
    "math500/test.jsonl": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl",
    "nltk/tokenizers/punkt.zip": f"{NLTK}/punkt.zip",
    "nltk/tokenizers/punkt_tab.zip": f"{NLTK}/punkt_tab.zip",
    # RULER v2 sources (evals/ruler2.py generates the sets from them)
    "ruler2/src/mmlu_test.parquet": f"{MMLU}/test-00000-of-00001.parquet",
    "ruler2/src/mmlu_auxiliary_train.parquet": f"{MMLU}/auxiliary_train-00000-of-00001.parquet",
    **{f"ruler2/src/hotpot_{n}.parquet": f"{HOTPOT}/{n}.parquet"
       for n in ("train-00000-of-00002", "train-00001-of-00002", "validation-00000-of-00001")},
    "long_context/tokenizer.json": "https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4/resolve/319f741cce68d7914884900c138a1fbb70a42f30/tokenizer.json",
}
# Needle corpus: CPython 3.12.7's Lib/*.py in path order, cut after the file that passes CORPUS_CHARS
# (about 800K tokens, enough for the default lengths 10K-262K on fresh slices).
CPYTHON = "https://codeload.github.com/python/cpython/tar.gz/c30085b0b5da62db9f0eba78e794772ff61b03cc"
CORPUS_CHARS = 3_000_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins() -> dict[str, str]:
    out = {}
    for line in PINS.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def check(names: list[str] | None = None) -> dict[str, str]:
    """sha256 of each named dataset file (all pinned files by default); exits on a missing or changed file."""
    pinned = pins()
    bad = []
    for name in names or sorted(pinned):
        path = ROOT / name
        if name not in pinned:
            bad.append(f"{name}: not pinned in {PINS.name}")
        elif not path.is_file():
            bad.append(f"{name}: missing (sharar-eval datasets fetch)")
        elif sha256(path) != pinned[name]:
            bad.append(f"{name}: sha256 differs from {PINS.name}")
    if bad:
        raise SystemExit("datasets under " + str(ROOT) + ":\n  " + "\n  ".join(bad))
    return {name: pinned[name] for name in names or sorted(pinned)}


def download(url: str) -> bytes:
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if "huggingface.co" in url and os.environ.get("HF_TOKEN") else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as r:
        return r.read()


def build_corpus() -> bytes:
    parts, size = [], 0
    with tarfile.open(fileobj=io.BytesIO(download(CPYTHON)), mode="r:gz") as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile() and m.name.endswith(".py") and "/Lib/" in m.name),
                         key=lambda m: m.name.split("/Lib/", 1)[1])
        for m in members:
            rel = m.name.split("/Lib/", 1)[1]
            text = tar.extractfile(m).read().decode("utf-8", errors="replace")
            parts.append(f"\n# ===== {rel} =====\n{text}")
            size += len(parts[-1])
            if size >= CORPUS_CHARS:
                break
    return "".join(parts).encode()


def fetch() -> None:
    for name, url in FILES.items():
        path = ROOT / name
        if path.is_file():
            continue
        print(f"fetch {name}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(download(url))
    he = ROOT / "accuracy/HumanEval.jsonl"
    if not he.is_file():
        he.write_bytes(gzip.decompress((ROOT / "accuracy/HumanEval.jsonl.gz").read_bytes()))
    for z in ("punkt", "punkt_tab"):
        if not (ROOT / "nltk/tokenizers" / z).is_dir():
            zipfile.ZipFile(ROOT / f"nltk/tokenizers/{z}.zip").extractall(ROOT / "nltk/tokenizers")
    corpus = ROOT / "long_context/corpus.txt"
    if not corpus.is_file():
        print("fetch long_context/corpus.txt (CPython 3.12.7 Lib)", flush=True)
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_bytes(build_corpus())
    from evals import ruler2  # generated from the pinned sources with the pinned tokenizer (about 15 minutes)
    for length in ruler2.LENGTHS.values():
        if not (ROOT / f"ruler2/ruler2-{length}.jsonl").is_file():
            print(f"generate ruler2/ruler2-{length}.jsonl", flush=True)
            ruler2.prepare(length)
    if not all((ROOT / "hard" / f).is_file() for f in ("mmlupro.jsonl", "gpqa_diamond.jsonl", "ifeval.jsonl", "wmt24pp.jsonl")):
        from evals import hard_eval  # needs pyarrow; GPQA is gated (HF_TOKEN)
        hard_eval.fetch()
    from evals import tbench  # Terminal-Bench needs harbor (uv tool install); skipped without it
    if tbench.HARBOR.is_file() and not (ROOT / tbench.MANIFEST).is_file():
        print("fetch tbench (harbor download)", flush=True)
        tbench.fetch()
    names = [n for n in pins() if tbench.HARBOR.is_file() or not n.startswith("tbench/")]
    check(names)
    print(f"{len(names)} pinned files match {PINS.name}" + ("" if len(names) == len(pins()) else " (Terminal-Bench skipped: no .venv-harbor)"))


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch()
    elif cmd == "check":
        print(f"all {len(check())} pinned files match {PINS.name}")
    else:
        print(__doc__)
        return 0 if cmd in ("-h", "--help") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
