#!/bin/bash
# Run RoboTwin evaluation for 10 tasks
# Each task connects to a dedicated server instance
# Results saved to eval_results/

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
ROBOTWIN_POLICY_CONFIG=${ROBOTWIN_POLICY_CONFIG:-/szeluresearch/fly/RoboTwin/policy/LingBotVA_B2/deploy_policy.yml}

SAVE_ROOT=${1:-"./eval_results/b2_step1200_$(date +%Y%m%d_%H%M%S)"}
TEST_NUM=${2:-20}          # episodes per task (use 100 for final evaluation)

# 10 tasks for evaluation, all using node 0 servers (127.0.0.1, ports 29056-29065)
# Node 0 has 8 GPUs (ports 29056-29063); 2 extra tasks use GPUs 0 and 1 (ports 29064-29065)
TASKS=(
    beat_block_hammer
    handover_block
    hanging_mug
    lift_pot
    click_bell
    move_can_pot
    pick_diverse_bottles
    stack_blocks_two
)
PORTS=(29056 29057 29058 29059 29060 29061 29062 29063)
HOSTS=(
    127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1
    127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1
)

mkdir -p "$SAVE_ROOT"
LOG_DIR="$SAVE_ROOT/client_logs"
mkdir -p "$LOG_DIR"

batch_time=$(date +%Y%m%d_%H%M%S)
pid_file="$LOG_DIR/client_pids_${batch_time}.txt"
> "$pid_file"

echo "=============================================="
echo "RoboTwin Evaluation - B2 Model (Step 1200)"
echo "Tasks: ${#TASKS[@]}  |  Episodes/task: $TEST_NUM"
echo "Save root: $SAVE_ROOT"
echo "=============================================="

for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    port="${PORTS[$i]}"
    host="${HOSTS[$i]}"
    gpu_id=$i   # Each client uses its own GPU (same as its server)
    log_file="${LOG_DIR}/${task}_${batch_time}.log"

    echo "Task $i: $task  GPU=$gpu_id HOST=$host PORT=$port  -> $log_file"

    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python -u -m evaluation.robotwin.eval_polict_client_openpi \
        --config "$ROBOTWIN_POLICY_CONFIG" \
        --overrides \
        --task_name "$task" \
        --task_config demo_clean \
        --ckpt_setting b2_step1200 \
        --seed 0 \
        --save_root "$SAVE_ROOT" \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --test_num "$TEST_NUM" \
        --host "$host" \
        --port "$port" > "$log_file" 2>&1 &

    pid=$!
    echo "$pid" >> "$pid_file"
    sleep 0.5
done

echo ""
echo "All ${#TASKS[@]} eval clients launched."
echo "PIDs saved to: $pid_file"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "When done, collect results:"
echo "  python scripts/collect_eval_results.py $SAVE_ROOT"
echo ""
echo "To stop all clients: kill \$(cat $pid_file)"

# Wait for all background jobs to finish
wait
echo ""
echo "All evaluations finished! Collecting results..."

# Print summary
python3 - <<'EOF'
import json, sys
from pathlib import Path

save_root = sys.argv[1] if len(sys.argv) > 1 else "."
results = []
for f in sorted(Path(save_root).rglob("*/metrics/*/result.json")):
    try:
        d = json.load(open(f))
        task = f.parts[-2]
        results.append((task, d.get("succ_rate", 0), d.get("total_num", 0)))
    except Exception:
        pass

if results:
    print("\n============ Evaluation Results ============")
    total_succ = 0
    total_tasks = len(results)
    for task, rate, num in results:
        print(f"  {task:35s}: {rate*100:.1f}%  ({int(rate*num)}/{int(num)})")
        total_succ += rate
    print(f"\n  Average success rate: {total_succ/total_tasks*100:.1f}%")
    print("============================================")
else:
    print("No result.json files found yet.")
EOF
