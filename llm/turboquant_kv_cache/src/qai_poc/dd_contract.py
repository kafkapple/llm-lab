"""Fail-closed validation for a proprietary quantizer evaluation contract."""

from collections.abc import Mapping

REQUIRED_FIELDS = (
    "method_version", "algorithm_spec", "target_tensors",
    "effective_bits_formula", "packing_format", "reference_decoder",
    "artifact_license", "benchmark_provenance",
)
VALID_TARGETS = frozenset({"K", "V", "embedding"})


def validate_dd_contract(contract: Mapping[str, object]) -> list[str]:
    """Return blocking errors; an empty list means the contract is testable."""
    errors = [
        f"missing or empty: {field}"
        for field in REQUIRED_FIELDS
        if not contract.get(field)
    ]
    targets = contract.get("target_tensors")
    if targets and (
        not isinstance(targets, list)
        or not set(targets).issubset(VALID_TARGETS)
    ):
        errors.append("target_tensors must be a list containing only K, V, embedding")
    claims = contract.get("claims", [])
    if not isinstance(claims, list):
        errors.append("claims must be a list")
    else:
        for index, claim in enumerate(claims):
            if not isinstance(claim, Mapping) or not all(
                claim.get(key) for key in ("statement", "metric", "raw_result")
            ):
                errors.append(
                    f"claims[{index}] needs statement, metric, and raw_result"
                )
    return errors


def is_testable(contract: Mapping[str, object]) -> bool:
    return not validate_dd_contract(contract)
