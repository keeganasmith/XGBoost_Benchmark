#!/bin/bash
module purge
module load WebProxy

cd "$SCRATCH/XGBoost_Benchmark" || exit 1
source modules.sh
source venv/bin/activate

export OMP_NUM_THREADS=96

mkdir -p composable_outputs

for ((i=0; i<3; i++)); do
  MEM_LOG="RAID_composable_outputs/composable_mem_usage_${i}.csv"

  # Header includes RAM + Swap
  echo "timestamp,mem_total_GB,mem_used_GB,mem_free_GB,mem_available_GB,swap_total_GB,swap_used_GB,swap_free_GB" > "$MEM_LOG"

  (
    while true; do
      ts=$(date +"%Y-%m-%d %H:%M:%S")

      # Use bytes for precision; convert to GB with decimals
      read mem_total_b mem_used_b mem_free_b mem_shared_b mem_buffcache_b mem_available_b < <(
        free -b | awk '/^Mem:/ {print $2,$3,$4,$5,$6,$7}'
      )

      read swap_total_b swap_used_b swap_free_b < <(
        free -b | awk '/^Swap:/ {print $2,$3,$4}'
      )

      # Convert bytes -> GB (decimal GB). If you prefer GiB, divide by 1024^3 instead.
      mem_total_gb=$(awk -v x="$mem_total_b" 'BEGIN{printf "%.3f", x/1e9}')
      mem_used_gb=$(awk -v x="$mem_used_b" 'BEGIN{printf "%.3f", x/1e9}')
      mem_free_gb=$(awk -v x="$mem_free_b" 'BEGIN{printf "%.3f", x/1e9}')
      mem_available_gb=$(awk -v x="$mem_available_b" 'BEGIN{printf "%.3f", x/1e9}')

      swap_total_gb=$(awk -v x="$swap_total_b" 'BEGIN{printf "%.3f", x/1e9}')
      swap_used_gb=$(awk -v x="$swap_used_b" 'BEGIN{printf "%.3f", x/1e9}')
      swap_free_gb=$(awk -v x="$swap_free_b" 'BEGIN{printf "%.3f", x/1e9}')

      echo "$ts,$mem_total_gb,$mem_used_gb,$mem_free_gb,$mem_available_gb,$swap_total_gb,$swap_used_gb,$swap_free_gb" >> "$MEM_LOG"
      sleep 2
    done
  ) &
  LOGGER_PID=$!

  bash run_script.sh > "RAID_composable_outputs/composable_output_${i}" 2>&1


  kill "$LOGGER_PID" 2>/dev/null
  wait "$LOGGER_PID" 2>/dev/null
done
