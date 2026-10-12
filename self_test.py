#!/usr/bin/env python3
"""Self-test for embedding-drift-watch. Run: python self_test.py"""
import json
import math
import os
import sys
import tempfile
import time
import contextlib
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_watch as dw

TESTS = []


def t(fn):
    TESTS.append(fn)
    return fn


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError("expected %s" % exc.__name__)


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = dw.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 0
    return code, out.getvalue(), err.getvalue()


TD = None


def setup():
    global TD
    TD = tempfile.mkdtemp(prefix="edw_test_")


def wp(name):
    return os.path.join(TD, name)


def wj(name, obj):
    p = wp(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return p


def wtxt(name, text):
    p = wp(name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def vecs(n=20, dim=4, shift=0.0, seed=7):
    x = seed
    out = []
    for _ in range(n):
        row = []
        for _ in range(dim):
            x = (x * 1103515245 + 12345) % (2 ** 31)
            row.append(((x % 2000) / 1000.0 - 1.0) + shift)
        out.append(row)
    return out


# ---------------- [A] functional

@t
def a01_fingerprint_shape():
    fp = dw.fingerprint(vecs(), buckets=4)
    for k in ("format_version", "n", "dim", "buckets", "mean_norm", "dims"):
        assert k in fp, k


@t
def a02_n_dim_recorded():
    fp = dw.fingerprint(vecs(n=17, dim=3))
    assert fp["n"] == 17 and fp["dim"] == 3


@t
def a03_hist_sums_to_n():
    fp = dw.fingerprint(vecs(n=25), buckets=8)
    for d in fp["dims"]:
        assert sum(d["hist"]) == 25


@t
def a04_identical_data_zero_drift():
    v = vecs()
    rep = dw.compare(dw.fingerprint(v), dw.fingerprint(v))
    assert rep["drift_score"] == 0.0


@t
def a05_identical_fp_compare():
    fp = dw.fingerprint(vecs())
    rep = dw.compare(fp, fp)
    assert rep["mean_tvd"] == 0.0 and rep["worst_tvd"] == 0.0


@t
def a06_shift_raises_score():
    base = dw.fingerprint(vecs(n=200))
    cur = dw.fingerprint(vecs(n=200, shift=0.9),
                         ranges=[d["range"] for d in base["dims"]])
    rep = dw.compare(base, cur)
    assert rep["drift_score"] > 0.3, rep


@t
def a07_worst_dim_index():
    v = vecs()
    cur = [list(r) for r in v]
    for r in cur:
        r[2] += 2.0
    base = dw.fingerprint(v)
    rep = dw.compare(base, dw.fingerprint(cur, ranges=[d["range"] for d in base["dims"]]))
    assert rep["worst_dim"] == 2, rep


@t
def a08_mean_shift_positive():
    rep = dw.compare(dw.fingerprint(vecs()), dw.fingerprint(vecs(shift=1.5)))
    assert rep["mean_shift"] > 0


@t
def a09_norm_drift_computed():
    base = dw.fingerprint(vecs())
    cur = dw.fingerprint([[x * 3.0 for x in r] for r in vecs()])
    assert dw.compare(base, cur)["norm_drift"] > 0


@t
def a10_cli_build():
    code, out, _ = run_cli(["build", "-i", wj("v.json", vecs()),
                            "-o", wp("b.json")])
    assert code == 0 and "baseline written" in out


@t
def a11_cli_check_similar_ok():
    run_cli(["build", "-i", wj("v.json", vecs(n=200)), "-o", wp("b.json")])
    code, out, _ = run_cli(["check", "-i", wj("v2.json", vecs(n=200, seed=9)),
                            "-b", wp("b.json"), "--threshold", "0.3"])
    assert code == 0 and "alert=false" in out, out


@t
def a12_cli_check_alert_exit1():
    run_cli(["build", "-i", wj("v.json", vecs(n=200)), "-o", wp("b.json")])
    code, out, _ = run_cli(["check", "-i", wj("v3.json", vecs(n=200, shift=2.0)),
                            "-b", wp("b.json")])
    assert code == 1 and "alert=true" in out, out


@t
def a13_cli_check_json_report():
    run_cli(["build", "-i", wj("v.json", vecs()), "-o", wp("b.json")])
    code, out, _ = run_cli(["check", "-i", wj("v.json", vecs()),
                            "-b", wp("b.json"), "--json"])
    rep = json.loads(out)
    for k in ("drift_score", "mean_tvd", "worst_dim", "alert", "threshold"):
        assert k in rep, k


@t
def a14_load_json_array():
    assert dw.load_vectors(wj("a.json", [[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


@t
def a15_load_json_wrapped():
    assert len(dw.load_vectors(wj("a.json", {"vectors": [[1.0]]}))) == 1


@t
def a16_load_jsonl_arrays():
    p = wtxt("a.jsonl", "[1,2]\n[3,4]\n")
    assert dw.load_vectors(p) == [[1, 2], [3, 4]]


@t
def a17_load_jsonl_dict():
    p = wtxt("a.jsonl", '{"vector": [1.0]}\n\n{"vector": [2.0]}\n')
    assert dw.load_vectors(p) == [[1.0], [2.0]]


@t
def a18_buckets_one():
    fp = dw.fingerprint(vecs(), buckets=1)
    assert all(d["hist"] == [len(vecs())] for d in fp["dims"])


@t
def a19_dim_one():
    fp = dw.fingerprint([[x] for x in range(10)])
    assert fp["dim"] == 1 and len(fp["dims"]) == 1


@t
def a20_norm_positive():
    fp = dw.fingerprint([[3.0, 4.0]] * 5)
    assert abs(fp["mean_norm"] - 5.0) < 1e-9

# ---------------- [B] exception / CLI paths

@t
def b01_empty_file():
    raises(ValueError, dw.load_vectors, wtxt("e.json", ""))


@t
def b02_whitespace_only():
    raises(ValueError, dw.load_vectors, wtxt("e.json", "  \n\t "))


@t
def b03_empty_jsonl():
    raises(ValueError, dw.load_vectors, wtxt("e.jsonl", "\n\n"))


@t
def b04_malformed_json_exit2():
    code, _, err = run_cli(["build", "-i", wtxt("e.json", "{bad"),
                            "-o", wp("o.json")])
    assert code == 2 and "error" in err


@t
def b05_malformed_jsonl_line():
    raises(ValueError, dw.load_vectors, wtxt("e.jsonl", "[1,2]\n{oops\n"))


@t
def b06_jsonl_bad_object():
    raises(ValueError, dw.load_vectors, wtxt("e.jsonl", '{"nope": 1}\n'))


@t
def b07_json_scalar_rejected():
    raises(ValueError, dw.load_vectors, wj("a.json", 42))


@t
def b08_dim_mismatch():
    raises(ValueError, dw.validate, [[1, 2], [1, 2, 3]])


@t
def b09_nan_rejected():
    raises(ValueError, dw.validate, [[1.0, float("nan")]])


@t
def b10_inf_rejected():
    raises(ValueError, dw.validate, [[float("inf")]])


@t
def b11_bool_rejected():
    raises(ValueError, dw.validate, [[True, False]])


@t
def b12_string_rejected():
    raises(ValueError, dw.validate, [["a", "b"]])


@t
def b13_empty_vector_rejected():
    raises(ValueError, dw.validate, [[]])


@t
def b14_no_vectors_rejected():
    raises(ValueError, dw.validate, [])


@t
def b15_missing_input_exit2():
    code, _, err = run_cli(["build", "-i", wp("missing.json"), "-o", wp("o.json")])
    assert code == 2 and "error" in err


@t
def b16_missing_baseline_exit2():
    code, _, _ = run_cli(["check", "-i", wj("v.json", vecs()),
                          "-b", wp("nobase.json")])
    assert code == 2


@t
def b17_corrupt_baseline_exit2():
    code, _, _ = run_cli(["check", "-i", wj("v.json", vecs()),
                          "-b", wtxt("b.json", "not json")])
    assert code == 2


@t
def b18_baseline_not_fingerprint_exit2():
    code, _, _ = run_cli(["check", "-i", wj("v.json", vecs()),
                          "-b", wj("b.json", {"foo": 1})])
    assert code == 2


@t
def b19_compare_dim_mismatch():
    raises(ValueError, dw.compare, dw.fingerprint(vecs(dim=3)),
           dw.fingerprint(vecs(dim=4)))


@t
def b20_bad_buckets():
    raises(ValueError, dw.fingerprint, vecs(), buckets=0)


@t
def b21_no_subcommand_exit2():
    code, _, _ = run_cli([])
    assert code == 2


@t
def b22_unknown_flag_exit2():
    code, _, _ = run_cli(["build", "--nope"])
    assert code == 2

# ---------------- [C] extreme edges

@t
def c01_single_vector():
    fp = dw.fingerprint([[1.0, 2.0]])
    assert fp["n"] == 1 and sum(fp["dims"][0]["hist"]) == 1
    assert max(fp["dims"][0]["hist"]) == 1


@t
def c02_constant_column():
    v = [[5.0, 1.0]] * 10
    fp = dw.fingerprint(v)
    d = fp["dims"][0]
    assert d["std"] == 0.0 and d["range"][1] > d["range"][0]
    rep = dw.compare(fp, dw.fingerprint(v))
    assert rep["drift_score"] == 0.0


@t
def c03_huge_dim():
    row = [0.001 * i for i in range(3000)]
    fp = dw.fingerprint([row, row])
    assert fp["dim"] == 3000


@t
def c04_large_values():
    v = [[1e6, -1e6]] * 8
    fp = dw.fingerprint(v)
    assert fp["dims"][0]["mean"] == 1e6


@t
def c05_tiny_values():
    v = [[1e-9, -1e-9]] * 8
    fp = dw.fingerprint(v)
    d = fp["dims"][0]
    assert sum(d["hist"]) == 8 and max(d["hist"]) == 8  # constant -> one bucket


@t
def c06_unicode_path():
    p = wp("向量_基线.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(vecs(n=5), f)
    code, _, _ = run_cli(["build", "-i", p, "-o", wp("b.json")])
    assert code == 0


@t
def c07_unicode_wrapped_key():
    assert len(dw.load_vectors(wj("a.json", {"vectors": [[1.0], [2.0]]}))) == 2


@t
def c08_many_vectors():
    v = vecs(n=400, dim=8)
    t0 = time.time()
    fp = dw.fingerprint(v)
    assert time.time() - t0 < 5.0 and fp["n"] == 400


@t
def c09_high_threshold_no_alert():
    run_cli(["build", "-i", wj("v.json", vecs()), "-o", wp("b.json")])
    code, _, _ = run_cli(["check", "-i", wj("v3.json", vecs(shift=2.0)),
                          "-b", wp("b.json"), "--threshold", "1000"])
    assert code == 0


@t
def c10_negative_threshold_alert():
    run_cli(["build", "-i", wj("v.json", vecs()), "-o", wp("b.json")])
    code, _, _ = run_cli(["check", "-i", wj("v.json", vecs()),
                          "-b", wp("b.json"), "--threshold", "-1"])
    assert code == 1


@t
def c11_buckets_64():
    fp = dw.fingerprint(vecs(), buckets=64)
    assert len(fp["dims"][0]["hist"]) == 64


@t
def c12_jsonl_blank_lines_skipped():
    p = wtxt("a.jsonl", "\n[1.0]\n   \n[2.0]\n")
    assert len(dw.load_vectors(p)) == 2


@t
def c13_roundtrip_report_serializable():
    rep = dw.compare(dw.fingerprint(vecs()), dw.fingerprint(vecs(shift=1.0)))
    json.loads(json.dumps(rep))


@t
def c14_std_consistency():
    col = [1.0, 2.0, 3.0]
    fp = dw.fingerprint([[x] for x in col])
    assert abs(fp["dims"][0]["std"] - math.sqrt(2.0 / 3.0)) < 1e-9


def main():
    setup()
    ok = 0
    fails = []
    for fn in TESTS:
        try:
            fn()
            ok += 1
        except Exception as e:  # noqa: BLE001
            fails.append((fn.__name__, repr(e)))
    for name, e in fails:
        print("FAIL %s: %s" % (name, e))
    total = len(TESTS)
    print("SUMMARY: %s %d/%d" % ("PASS" if not fails else "FAIL", ok, total))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
