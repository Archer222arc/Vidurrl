#!/bin/bash
# Quick test script to validate PPO improvements with a short training run

echo "🚀 Testing Improved PPO Scheduler"
echo "=================================="

# Configuration
NUM_REQUESTS=100
QPS=2
NUM_REPLICAS=4
OUTPUT_DIR="./outputs/quick_test_improved_ppo/run_$(date +%Y%m%d_%H%M%S)"

echo "📊 Test Configuration:"
echo "  - Requests: $NUM_REQUESTS"
echo "  - QPS: $QPS"
echo "  - Replicas: $NUM_REPLICAS"
echo "  - Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run quick PPO test
echo "🔧 Running improved PPO scheduler..."

python -m vidur \
  --cluster_config_num_replicas $NUM_REPLICAS \
  --global_scheduler_config_type ppo_modular \
  --global_scheduler_config_enable_tensorboard false \
  --global_scheduler_config_metrics_export_enabled true \
  --global_scheduler_config_metrics_export_path "$OUTPUT_DIR/metrics" \
  --global_scheduler_config_checkpoint_dir "$OUTPUT_DIR/checkpoints" \
  --global_scheduler_config_enable_checkpoints false \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests $NUM_REQUESTS \
  --request_interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps $QPS \
  --output_dir "$OUTPUT_DIR/logs" \
  > "$OUTPUT_DIR/ppo_output.log" 2>&1

# Check if the run completed successfully
if [ $? -eq 0 ]; then
    echo "✅ PPO test completed successfully!"

    # Analyze the output for key metrics
    echo ""
    echo "📈 Analyzing results..."

    # Extract reward statistics from the log
    if [ -f "$OUTPUT_DIR/ppo_output.log" ]; then
        echo "🎯 Reward Analysis:"
        grep -E "last_r=" "$OUTPUT_DIR/ppo_output.log" | tail -5 | while read line; do
            echo "  $line"
        done

        echo ""
        echo "🔄 Action Distribution Analysis:"
        grep -E "act=\[" "$OUTPUT_DIR/ppo_output.log" | tail -3 | while read line; do
            echo "  $line"
        done

        echo ""
        echo "📊 Training Update Analysis:"
        grep -E "\[PPO:update\]" "$OUTPUT_DIR/ppo_output.log" | tail -2 | while read line; do
            echo "  $line"
        done
    fi

    # Check for zero rewards (should be much fewer now)
    zero_reward_count=$(grep -c "last_r=0.000000" "$OUTPUT_DIR/ppo_output.log" 2>/dev/null || echo "0")
    total_steps=$(grep -c "last_r=" "$OUTPUT_DIR/ppo_output.log" 2>/dev/null || echo "1")

    if [ "$total_steps" -gt "0" ]; then
        zero_percentage=$(echo "scale=1; $zero_reward_count * 100 / $total_steps" | bc -l 2>/dev/null || echo "N/A")
        echo ""
        echo "🎯 Reward Density Check:"
        echo "  Zero rewards: $zero_reward_count/$total_steps steps ($zero_percentage%)"

        if [ "$zero_reward_count" -lt "$(echo "$total_steps * 0.5" | bc -l | cut -d. -f1)" ]; then
            echo "  ✅ GOOD: Much fewer zero rewards than before (was ~99%)"
        else
            echo "  ⚠️  Still many zero rewards, may need further tuning"
        fi
    fi

    echo ""
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo "📄 Full log: $OUTPUT_DIR/ppo_output.log"

else
    echo "❌ PPO test failed! Check logs:"
    tail -20 "$OUTPUT_DIR/ppo_output.log"
    exit 1
fi

echo ""
echo "🏁 Test completed. Key improvements to look for:"
echo "   1. Fewer zero rewards (target: <50% vs previous 99%)"
echo "   2. More balanced action distribution (closer to [25,25,25,25])"
echo "   3. Higher reward variance in training updates"
echo "   4. Better entropy values (around 1.3 for exploration)"