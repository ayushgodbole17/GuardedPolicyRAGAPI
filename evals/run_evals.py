import argparse
import asyncio
import json
import os
import httpx

GOLDEN_SET = [
    {
        "question": "How many days of PTO do employees get in their first two years?",
        "expected_keywords": ["15", "days"],
        "should_refuse": False,
    },
    {
        "question": "What is the maximum hotel rate allowed in New York City?",
        "expected_keywords": ["300"],
        "should_refuse": False,
    },
    {
        "question": "Is multi-factor authentication required for remote access?",
        "expected_keywords": ["mfa", "multi-factor", "required", "mandatory"],
        "should_refuse": False,
    },
    {
        "question": "How much parental leave do primary caregivers receive?",
        "expected_keywords": ["16", "weeks"],
        "should_refuse": False,
    },
    {
        "question": "What is the daily meal per diem limit when travelling?",
        "expected_keywords": ["105", "per diem"],
        "should_refuse": False,
    },
    {
        "question": "What is the current stock price of Acme Corporation?",
        "expected_keywords": [],
        "should_refuse": True,
    },
]


def _run_ragas(ragas_samples: list[dict]) -> dict[str, float]:
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy
    except ImportError:
        print("\n[RAGAS] not installed — run: pip install -r requirements-evals.txt\n")
        return {}

    samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
        )
        for s in ragas_samples
    ]

    dataset = EvaluationDataset(samples=samples)
    print("\nRunning RAGAS evaluation...")
    result = evaluate(dataset=dataset, metrics=[Faithfulness(), AnswerRelevancy()])
    return result.to_pandas()[["faithfulness", "answer_relevancy"]].mean().to_dict()


async def run(base_url: str, skip_ragas: bool) -> None:
    results = []
    passed = 0
    ragas_samples = []

    print(f"\n{'Question':<62} {'Pass':<6} {'Confidence':<12} {'Latency'}")
    print("-" * 95)

    async with httpx.AsyncClient(timeout=60) as client:
        for item in GOLDEN_SET:
            resp = await client.post(
                f"{base_url}/ask",
                json={"question": item["question"]},
            )
            resp.raise_for_status()
            data = resp.json()

            refused = data["refused"]
            answer = data.get("answer", "")
            confidence = data.get("confidence", 0.0)
            latency_ms = data.get("latency_ms", 0)
            contexts = [h["snippet"] for h in data.get("hits", [])] or [""]

            if item["should_refuse"]:
                ok = refused
            else:
                keywords_found = all(
                    kw.lower() in answer.lower() for kw in item["expected_keywords"]
                )
                ok = not refused and keywords_found

            if ok:
                passed += 1

            label = "PASS" if ok else "FAIL"
            q = item["question"][:60] + ".." if len(item["question"]) > 60 else item["question"]
            print(f"{q:<62} {label:<6} {confidence:<12.4f} {latency_ms}ms")

            result_entry = {
                "question": item["question"],
                "passed": ok,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "answer": answer,
                "refused": refused,
            }

            if not item["should_refuse"] and not refused:
                ragas_samples.append({
                    "question": item["question"],
                    "answer": answer,
                    "contexts": contexts,
                })

            results.append(result_entry)

    print("-" * 95)
    print(f"\n{passed}/{len(GOLDEN_SET)} passed\n")

    ragas_scores = {}
    if not skip_ragas and ragas_samples:
        ragas_scores = _run_ragas(ragas_samples)
        if ragas_scores:
            print("\nRAGAS scores (mean across answered questions):")
            for metric, score in ragas_scores.items():
                print(f"  {metric:<25} {score:.4f}")
            print()

    out = {
        "summary": {
            "passed": passed,
            "total": len(GOLDEN_SET),
            "pass_rate": passed / len(GOLDEN_SET),
            **ragas_scores,
        },
        "cases": results,
    }

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.skip_ragas))
