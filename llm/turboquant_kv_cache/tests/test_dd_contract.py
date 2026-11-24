from qai_poc.dd_contract import is_testable, validate_dd_contract


def complete_contract():
    return {
        "method_version": "example-v1",
        "algorithm_spec": "spec.pdf#sha256=example",
        "target_tensors": ["K", "V"],
        "effective_bits_formula": "(payload + metadata + padding) / values",
        "packing_format": "format.md",
        "reference_decoder": "decoder.py#sha256=example",
        "artifact_license": "internal-evaluation",
        "benchmark_provenance": "run-manifest.json",
        "claims": [{
            "statement": "example only",
            "metric": "measured bytes/token",
            "raw_result": "raw.json",
        }],
    }


def test_complete_contract_is_testable():
    assert is_testable(complete_contract())


def test_missing_decoder_fails_closed():
    contract = complete_contract()
    contract["reference_decoder"] = ""
    assert "missing or empty: reference_decoder" in validate_dd_contract(contract)


def test_claim_requires_metric_and_raw_result():
    contract = complete_contract()
    contract["claims"] = [{"statement": "10x"}]
    assert validate_dd_contract(contract) == [
        "claims[0] needs statement, metric, and raw_result"
    ]
