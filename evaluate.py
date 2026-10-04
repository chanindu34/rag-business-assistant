"""
Evaluate the assistant against eval/questions.yaml.

    python3 evaluate.py --check            # verify the answer key against the report, no API calls
    python3 evaluate.py                    # full run (~2-3 API calls per question)
    python3 evaluate.py --judge            # also check every answered claim against its sources
    python3 evaluate.py --only ebitda_group,key_risks

Metrics
  retrieval hit     a passage from a correct page reached the answer model (answerable questions)
  answer correct    every expected figure / term appears in the answer, and it was not declined
  correct decline   unanswerable questions were declined; answerable ones were not
  faithfulness      (--judge) an independent model finds no claim unsupported by the cited passages
  latency           retrieval and end-to-end, median and 95th percentile

Results go to eval/results/<timestamp>.json and .md.
"""

import argparse
import json
import logging
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from config import COLLECTION_NAME, DEFAULT_TOP_K, HIGH_THRESHOLD, JUDGE_FALLBACKS, JUDGE_MODEL, LOW_THRESHOLD, parents_path
from pipeline import REFUSAL_TOKEN, build_prompt, sources_from_result

ROOT = Path(__file__).parent
QUESTIONS = ROOT / "eval" / "questions.yaml"
RESULTS = ROOT / "eval" / "results"


# ---------------------------------------------------------------------------
# Matching expected values in free text
# ---------------------------------------------------------------------------
def normalise(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\*\*|__|`", "", t)                     # markdown emphasis
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)           # 3,468 -> 3468
    t = re.sub(r"\s*(?:per cent|percent)\b", "%", t)
    t = re.sub(r"(?<=\d)\s+%", "%", t)
    t = re.sub(r"\brs\.?\s*(?=\d)", "rs ", t)            # Rs.35.72 / Rs. 35.72 -> rs 35.72
    return re.sub(r"\s+", " ", t)


def contains(text: str, needle: str) -> bool:
    """Number-aware match: "5%" must not match inside "5.4%" or "35%"."""
    t, n = normalise(text), normalise(needle)
    if re.search(r"\d", n):
        return re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", t) is not None
    return n in t


def expected_found(text: str, q: dict):
    missing = [m for m in q.get("must_include", []) if not contains(text, m)]
    any_of = q.get("any_of", [])
    any_ok = not any_of or any(contains(text, a) for a in any_of)
    if not any_ok:
        missing.append("one of " + " / ".join(any_of))
    return not missing, missing


def load_questions():
    qs = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids)), "duplicate ids in questions.yaml"
    return qs


# ---------------------------------------------------------------------------
# --check: validate the answer key against the report (no API)
# ---------------------------------------------------------------------------
def check_answer_key(qs) -> int:
    parents = json.loads(Path(parents_path(COLLECTION_NAME)).read_text(encoding="utf-8"))
    by_page = {}
    for p in parents.values():
        by_page.setdefault(p["page"], []).append(p["text"])
    problems = 0
    for q in qs:
        if q["type"] == "unanswerable":
            continue
        page_text = " ".join(t for pg in q["pages"] for t in by_page.get(pg, []))
        missing_pages = [pg for pg in q["pages"] if pg not in by_page]
        ok, missing = expected_found(page_text, q)
        if missing_pages or not ok:
            problems += 1
            print(f"  PROBLEM {q['id']}: pages not indexed {missing_pages}  missing on pages {missing}")
    counts = {}
    for q in qs:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    print(f"{len(qs)} questions: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    print("Answer key OK: every expected value appears on its listed page." if not problems
          else f"{problems} question(s) need fixing.")
    return problems


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are checking an answer for faithfulness to its sources.

<sources>
{sources}
</sources>

<answer>
{answer}
</answer>

List every factual claim in the answer that is NOT supported by the sources.
Reply with JSON only: {{"unsupported": ["claim", ...]}}. Use an empty list if every claim is supported."""


def percentile(values, p):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def run(qs, use_judge: bool, use_cache: bool):
    from llm import GeminiGateway, QuotaExhaustedError
    from pipeline import build_rag, load_index, make_client, make_gateways
    from query_condensation import QueryCondenser

    client = make_client()
    answers, hyde = make_gateways(client, use_cache=use_cache)
    rag = build_rag(client, load_index(),
                    hyde_generate_fn=lambda p: hyde.generate(p, purpose="hyde", max_output_tokens=200, temperature=0.7))
    judge = GeminiGateway(client, [JUDGE_MODEL] + JUDGE_FALLBACKS) if use_judge else None
    condenser = QueryCondenser()

    rows = []
    for i, q in enumerate(qs, 1):
        history = []
        for h in q.get("history", []):
            resolved, _ = condenser.resolve(history, h)
            history.append({"role": "user", "content": h, "resolved_query": resolved})
        resolved, previous = condenser.resolve(history, q["question"])

        t0 = time.time()
        result = rag.retrieve(resolved, top_k=DEFAULT_TOP_K)
        t_retrieve = time.time() - t0
        sources = sources_from_result(result["final_chunks"])
        stats = result["retrieval_stats"]

        answer_text, declined, error = "", False, None
        if result["confidence"] == "low":
            declined = True
        else:
            try:
                answer_text = answers.generate(build_prompt(q["question"], sources, previous), purpose="answer")
                declined = answer_text.strip().startswith(REFUSAL_TOKEN)
            except QuotaExhaustedError as e:
                error = str(e)
        t_total = time.time() - t0

        answerable = q["type"] != "unanswerable"
        pages = sorted({s["page"] for s in sources if s.get("page")})
        row = {
            "id": q["id"], "type": q["type"], "question": q["question"], "answerable": answerable,
            "route": stats.get("route"), "hyde_used": stats.get("hyde_used"),
            "top_score": stats.get("top_score"), "tier": result["confidence"],
            "retrieved_pages": pages, "gold_pages": q.get("pages", []),
            "declined": declined, "answer": answer_text, "error": error,
            "retrieve_s": round(t_retrieve, 2), "total_s": round(t_total, 2),
        }
        if answerable:
            row["retrieval_hit"] = bool(set(pages) & set(q["pages"]))
            ok, missing = expected_found(answer_text, q)
            row["answer_correct"] = ok and not declined and not error
            row["missing"] = missing
            row["decision_correct"] = not declined
        else:
            row["decision_correct"] = declined

        if judge and answer_text and not declined and not error:
            try:
                raw = judge.generate(JUDGE_PROMPT.format(
                    sources="\n\n".join(f"[{k}] (page {s.get('page')}) {s['text']}" for k, s in enumerate(sources, 1)),
                    answer=answer_text), purpose="judge", temperature=0.0)
                m = re.search(r"\{.*\}", raw, re.S)
                unsupported = json.loads(m.group(0)).get("unsupported", []) if m else None
                row["unsupported_claims"] = unsupported
                row["faithful"] = unsupported == [] if unsupported is not None else None
            except QuotaExhaustedError as e:
                row["judge_error"] = str(e)
            except (ValueError, AttributeError):
                row["faithful"] = None

        mark = "OK " if row["decision_correct"] and row.get("answer_correct", True) else "XX "
        print(f"{mark}[{i:>2}/{len(qs)}] {q['id']:<28} tier={row['tier']:<9} score={row['top_score']:.2f} "
              f"pages={pages[:6]} {('MISSING ' + str(row.get('missing'))) if row.get('missing') else ''}")
        rows.append(row)
    return rows


def summarise(rows):
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    rate = lambda xs, key: (sum(1 for x in xs if x.get(key)) / len(xs)) if xs else None
    judged = [r for r in rows if r.get("faithful") is not None]
    by_type = {}
    for r in ans:
        d = by_type.setdefault(r["type"], {"n": 0, "correct": 0, "hit": 0})
        d["n"] += 1
        d["correct"] += bool(r.get("answer_correct"))
        d["hit"] += bool(r.get("retrieval_hit"))
    scores_ans = sorted(r["top_score"] for r in ans if r.get("top_score") is not None)
    scores_una = sorted(r["top_score"] for r in una if r.get("top_score") is not None)
    return {
        "questions": len(rows),
        "answerable": len(ans),
        "unanswerable": len(una),
        "retrieval_hit_rate": rate(ans, "retrieval_hit"),
        "answer_accuracy": rate(ans, "answer_correct"),
        "false_declines": sum(1 for r in ans if r["declined"]),
        "correct_declines": sum(1 for r in una if r["declined"]),
        "faithfulness": (sum(1 for r in judged if r["faithful"]) / len(judged)) if judged else None,
        "judged": len(judged),
        "by_type": by_type,
        "latency_retrieve_p50": percentile([r["retrieve_s"] for r in rows], 50),
        "latency_retrieve_p95": percentile([r["retrieve_s"] for r in rows], 95),
        "latency_total_p50": percentile([r["total_s"] for r in rows], 50),
        "latency_total_p95": percentile([r["total_s"] for r in rows], 95),
        "hyde_rate": rate(rows, "hyde_used"),
        "rerank_score_answerable": scores_ans,
        "rerank_score_unanswerable": scores_una,
        "thresholds": {"high": HIGH_THRESHOLD, "low": LOW_THRESHOLD},
    }


def write_report(rows, summary):
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    (RESULTS / f"{stamp}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"
    sec = lambda x: "n/a" if x is None else f"{x:.1f}s"
    lines = [
        f"# Evaluation {stamp}", "",
        f"{summary['questions']} questions ({summary['answerable']} answerable, {summary['unanswerable']} unanswerable)", "",
        "| Metric | Result |", "|---|---|",
        f"| Retrieval hit rate (a correct page reached the model) | {pct(summary['retrieval_hit_rate'])} |",
        f"| Answer accuracy (all expected figures present) | {pct(summary['answer_accuracy'])} |",
        f"| Unanswerable questions correctly declined | {summary['correct_declines']} / {summary['unanswerable']} |",
        f"| Answerable questions wrongly declined | {summary['false_declines']} / {summary['answerable']} |",
        f"| Faithfulness (independent judge, {summary['judged']} answers) | {pct(summary['faithfulness'])} |",
        f"| Retrieval latency, median / p95 | {sec(summary['latency_retrieve_p50'])} / {sec(summary['latency_retrieve_p95'])} |",
        f"| End-to-end latency, median / p95 | {sec(summary['latency_total_p50'])} / {sec(summary['latency_total_p95'])} |",
        f"| Questions routed through HyDE | {pct(summary['hyde_rate'])} |", "",
        "| Type | Questions | Retrieval hit | Answer correct |", "|---|---|---|---|",
    ]
    for t, d in summary["by_type"].items():
        lines.append(f"| {t} | {d['n']} | {d['hit']}/{d['n']} | {d['correct']}/{d['n']} |")
    lines += ["", "## Failures", ""]
    for r in rows:
        if not r["decision_correct"] or (r["answerable"] and not r.get("answer_correct")):
            reason = ("wrongly declined" if r["answerable"] and r["declined"]
                      else "answered an unanswerable question" if not r["answerable"]
                      else f"missing {r.get('missing')}")
            lines.append(f"- **{r['id']}**: {reason}. Retrieved pages {r['retrieved_pages']}, expected {r.get('gold_pages')}.")
    lines += ["", "## Reranker top score: answerable vs unanswerable", "",
              f"- Answerable: {[round(s, 2) for s in summary['rerank_score_answerable']]}",
              f"- Unanswerable: {[round(s, 2) for s in summary['rerank_score_unanswerable']]}"]
    path = RESULTS / f"{stamp}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the answer key against the report, no API calls")
    ap.add_argument("--judge", action="store_true", help="also run the faithfulness judge (+1 API call per answer)")
    ap.add_argument("--fresh", action="store_true", help="ignore the answer cache")
    ap.add_argument("--only", help="comma-separated question ids")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    qs = load_questions()
    if args.only:
        wanted = set(args.only.split(","))
        qs = [q for q in qs if q["id"] in wanted]
    if args.check:
        sys.exit(1 if check_answer_key(qs) else 0)

    if check_answer_key(qs):
        sys.exit("Fix the answer key before running the evaluation.")
    print()
    rows = run(qs, use_judge=args.judge, use_cache=not args.fresh)
    summary = summarise(rows)
    path = write_report(rows, summary)
    print("\n" + path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
