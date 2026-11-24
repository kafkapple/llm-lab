"""Compare two run_vllm_logprobs.py dumps: per-position top-K KL(reference || candidate).

Missing-mass caveat: KL is computed over the union of each position's top-K
token ids. A token absent from a distribution's own top-K is floored to
exp(min_logprob_in_that_distribution) rather than 0, avoiding infinite KL;
this underestimates true tail probability and is a known approximation
(same simplification as the oobabooga localbench measurement in F5).
"""

import argparse
import json
import math
from pathlib import Path


def position_kl(ref: dict[str, float], cand: dict[str, float]) -> float:
    ref_floor = min(ref.values())
    cand_floor = min(cand.values())
    support = set(ref) | set(cand)
    kl = 0.0
    for token_id in support:
        p = math.exp(ref.get(token_id, ref_floor))
        q = math.exp(cand.get(token_id, cand_floor))
        kl += p * (math.log(p) - math.log(q))
    return kl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True, help="native-dtype logprobs dump")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    cand_by_id = {r["id"]: r for r in candidate["records"]}
    per_record = []
    for ref_record in reference["records"]:
        cand_record = cand_by_id.get(ref_record["id"])
        if cand_record is None or len(cand_record["positions"]) != len(ref_record["positions"]):
            continue  # generation diverged in length; not comparable position-by-position
        kls = [
            position_kl(rp, cp)
            for rp, cp in zip(ref_record["positions"], cand_record["positions"])
        ]
        per_record.append({"id": ref_record["id"], "mean_kl": sum(kls) / len(kls) if kls else None})
    valid = [r["mean_kl"] for r in per_record if r["mean_kl"] is not None]
    payload = {
        "reference_kv_dtype": reference["kv_dtype"],
        "reference_skip_layers_resolved": reference["kv_skip_layers_resolved"],
        "candidate_kv_dtype": candidate["kv_dtype"],
        "candidate_skip_layers_resolved": candidate["kv_skip_layers_resolved"],
        "n_records_total": len(reference["records"]),
        "n_records_comparable": len(valid),
        "n_records_length_mismatch": len(reference["records"]) - len(valid),
        "mean_kl": sum(valid) / len(valid) if valid else None,
        "records": per_record,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}))


if __name__ == "__main__":
    main()
