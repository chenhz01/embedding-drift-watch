#!/usr/bin/env python3
"""embedding-drift-watch: distribution fingerprinting + drift alerts for embedding sets.

Pure standard library. Build a statistical fingerprint (per-dimension mean/std +
bucketed histogram, plus mean L2 norm) of a vector set, then compare a new set
against the baseline and alert when aggregate drift exceeds a threshold.
"""
import argparse
import json
import math
import sys

FORMAT_VERSION = 1


# ---------------------------------------------------------------- input

def _num(x):
    """Strict finite-number check; raises ValueError otherwise."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError("non-numeric vector component: %r" % (x,))
    f = float(x)
    if math.isnan(f) or math.isinf(f):
        raise ValueError("non-finite vector component: %r" % (x,))
    return f


def load_vectors(path):
    """Load vectors from .json (array, or {"vectors": [...]}) or .jsonl.

    Each JSONL line is either an array or {"vector": [...]}.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ValueError("empty input: no vectors found in %s" % path)
    if path.endswith(".jsonl"):
        vecs = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError("line %d: invalid JSON: %s" % (i, e))
            if isinstance(obj, list):
                vecs.append(obj)
            elif isinstance(obj, dict) and "vector" in obj:
                vecs.append(obj["vector"])
            else:
                raise ValueError("line %d: expected array or {\"vector\": [...]}" % i)
        return vecs
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError("invalid JSON in %s: %s" % (path, e))
    if isinstance(data, dict) and "vectors" in data:
        data = data["vectors"]
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of vectors")
    return data


def validate(vectors):
    """Return dim; raise ValueError on empty / ragged / non-finite vectors."""
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("no vectors provided")
    dim = None
    for i, v in enumerate(vectors):
        if not isinstance(v, list) or not v:
            raise ValueError("vector %d: not a non-empty array" % i)
        if dim is None:
            dim = len(v)
        elif len(v) != dim:
            raise ValueError("vector %d: dimension %d != %d" % (i, len(v), dim))
        for x in v:
            _num(x)
    return dim


# ---------------------------------------------------------------- fingerprint

def fingerprint(vectors, buckets=8, ranges=None):
    """Build a distribution fingerprint of a vector set.

    Per dimension: mean, std, a bucketed histogram. By default the histogram
    range is derived from the data itself; pass ``ranges`` (one [lo, hi] pair
    per dimension, e.g. from a baseline fingerprint) to bucket values on a
    fixed reference range so pure distribution shifts are visible.
    Global: mean L2 norm of the vectors.
    """
    if not isinstance(buckets, int) or buckets < 1:
        raise ValueError("buckets must be a positive integer")
    dim = validate(vectors)
    if ranges is not None:
        if not isinstance(ranges, list) or len(ranges) != dim:
            raise ValueError("ranges must be a list with one [lo, hi] per dimension")
    n = len(vectors)
    cols = [[float(v[d]) for v in vectors] for d in range(dim)]
    dims = []
    for d, col in enumerate(cols):
        mean = sum(col) / n
        var = sum((x - mean) ** 2 for x in col) / n
        std = math.sqrt(var)
        if ranges is not None:
            lo, hi = float(ranges[d][0]), float(ranges[d][1])
        else:
            lo, hi = min(col), max(col)
            if hi <= lo:  # constant column: give the range a unit width
                lo, hi = lo - 0.5, hi + 0.5
        width = (hi - lo) / buckets
        hist = [0] * buckets
        for x in col:
            b = int((x - lo) / width)
            if b < 0:
                b = 0
            elif b >= buckets:
                b = buckets - 1
            hist[b] += 1
        dims.append({"mean": mean, "std": std, "range": [lo, hi], "hist": hist})
    norms = [math.sqrt(sum(x * x for x in (float(c) for c in v))) for v in vectors]
    return {
        "format_version": FORMAT_VERSION,
        "n": n,
        "dim": dim,
        "buckets": buckets,
        "mean_norm": sum(norms) / n,
        "dims": dims,
    }


# ---------------------------------------------------------------- compare

def _tvd(h1, h2):
    n1, n2 = sum(h1), sum(h2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return 0.5 * sum(abs(a / n1 - b / n2) for a, b in zip(h1, h2))


def compare(base, cur):
    """Compare two fingerprints; returns aggregate drift report.

    drift_score = 0.7 * mean TVD + 0.2 * min(mean_shift, 1)
                  + 0.1 * min(norm_drift, 1)
    """
    if base.get("dim") != cur.get("dim"):
        raise ValueError("dimension mismatch: baseline %s vs current %s"
                         % (base.get("dim"), cur.get("dim")))
    if not base.get("dims") or not cur.get("dims"):
        raise ValueError("fingerprint has no per-dimension data")
    tvds = [_tvd(b["hist"], c["hist"]) for b, c in zip(base["dims"], cur["dims"])]
    mean_tvd = sum(tvds) / len(tvds)
    shifts = []
    for b, c in zip(base["dims"], cur["dims"]):
        denom = b["std"] if b["std"] > 1e-12 else 1.0
        shifts.append(abs(b["mean"] - c["mean"]) / denom)
    mean_shift = sum(shifts) / len(shifts)
    bn = float(base.get("mean_norm") or 0.0)
    cn = float(cur.get("mean_norm") or 0.0)
    norm_drift = abs(bn - cn) / (bn + 1e-12)
    score = 0.7 * mean_tvd + 0.2 * min(mean_shift, 1.0) + 0.1 * min(norm_drift, 1.0)
    worst = max(tvds)
    return {
        "drift_score": round(score, 6),
        "mean_tvd": round(mean_tvd, 6),
        "worst_tvd": round(worst, 6),
        "worst_dim": tvds.index(worst),
        "mean_shift": round(mean_shift, 6),
        "norm_drift": round(norm_drift, 6),
    }


# ---------------------------------------------------------------- CLI

def cmd_build(args):
    vecs = load_vectors(args.input)
    fp = fingerprint(vecs, buckets=args.buckets)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fp, f)
    print("baseline written: n=%d dim=%d buckets=%d -> %s"
          % (fp["n"], fp["dim"], fp["buckets"], args.output))
    return 0


def cmd_check(args):
    with open(args.baseline, "r", encoding="utf-8") as f:
        base = json.load(f)
    if not isinstance(base, dict) or "dims" not in base:
        raise ValueError("baseline file is not a fingerprint")
    base_ranges = [d["range"] for d in base["dims"]]
    cur = fingerprint(load_vectors(args.input), buckets=base.get("buckets", 8),
                      ranges=base_ranges)
    rep = compare(base, cur)
    rep["threshold"] = args.threshold
    rep["alert"] = rep["drift_score"] > args.threshold
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print("drift_score=%s mean_tvd=%s worst_dim=%d alert=%s"
              % (rep["drift_score"], rep["mean_tvd"], rep["worst_dim"],
                 str(rep["alert"]).lower()))
    return 1 if rep["alert"] else 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="drift-watch",
        description="Embedding distribution fingerprinting and drift alerts.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a baseline fingerprint")
    b.add_argument("-i", "--input", required=True)
    b.add_argument("-o", "--output", required=True)
    b.add_argument("--buckets", type=int, default=8)
    b.set_defaults(fn=cmd_build)

    c = sub.add_parser("check", help="compare a vector set against a baseline")
    c.add_argument("-i", "--input", required=True)
    c.add_argument("-b", "--baseline", required=True)
    c.add_argument("--threshold", type=float, default=0.2)
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_check)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
