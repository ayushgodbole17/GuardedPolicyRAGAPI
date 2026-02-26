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


async def run(base_url: str) -> None:
    results = []
    passed = 0

    print(f"\n{'Question':<62} {'Pass':<6} {'Confidence':<12} {'Latency'}")
    print("-" * 95)

    async with httpx.AsyncClient(timeout=30) as client:
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

            results.append({
                "question": item["question"],
                "passed": ok,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "answer": answer,
            })

    print("-" * 95)
    print(f"\n{passed}/{len(GOLDEN_SET)} passed\n")

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    asyncio.run(run(args.base_url))
