#!/usr/bin/env bash
set -euo pipefail

output="${1:-results/gpu_monitor.csv}"
interval="${2:-10}"
gpu="${3:-}"
max_min="${4:-720}"   # a sampler with no end outlives the run it was started for
mkdir -p "$(dirname "$output")"
echo "timestamp,index,memory_used_mib,utilization_pct,temperature_c,power_w" > "$output"
end=$(( $(date +%s) + max_min * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  args=()
  if [[ -n "$gpu" ]]; then
    args+=("--id=$gpu")
  fi
  nvidia-smi "${args[@]}" \
    --query-gpu=timestamp,index,memory.used,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits >> "$output"
  sleep "$interval"
done
echo "monitor_gpu: stopped after ${max_min} min" >&2
