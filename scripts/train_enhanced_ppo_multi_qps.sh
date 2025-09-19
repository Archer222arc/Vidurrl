#!/bin/bash

# Enhanced PPO Multi-QPS Training Script
# Tests the improved PPO scheduler with enhanced features across different QPS scenarios

set -e

# Configuration
PROJECT_ROOT="/Users/ruicheng/Documents/GitHub/Vidur/Vidur_arc2"
PYTHONPATH="$PROJECT_ROOT"

# Enhanced PPO parameters
ENABLE_ENHANCED_FEATURES=True
ENABLE_DYNAMIC_TEMPERATURE=True
METRICS_EXPORT_ENABLED=True

# Reward enhancement parameters
LATENCY_THRESHOLD=2.0
LATENCY_PENALTY_SCALE=5.0
LOAD_BALANCE_PENALTY=0.03

# Temperature control parameters
BASE_TEMPERATURE=1.0
MIN_TEMPERATURE=0.5
MAX_TEMPERATURE=2.0
QPS_SENSITIVITY=0.1
LATENCY_SENSITIVITY=0.2

# State enhancement parameters
STATE_HISTORY_WINDOW=5
QPS_WINDOW=10

echo "🚀 Enhanced PPO Multi-QPS Training & Validation"
echo "================================================"
echo "Enhanced Features: $ENABLE_ENHANCED_FEATURES"
echo "Dynamic Temperature: $ENABLE_DYNAMIC_TEMPERATURE"
echo "Metrics Export: $METRICS_EXPORT_ENABLED"
echo ""

# Test scenarios
QPS_SCENARIOS=(1.5 2.0 2.5)
STEPS_PER_SCENARIO=300

for qps in "${QPS_SCENARIOS[@]}"; do
    echo "📊 Training QPS $qps scenario..."

    # Create scenario-specific output directory
    SCENARIO_DIR="outputs/runs/enhanced_ppo/qps_${qps}"
    mkdir -p "$SCENARIO_DIR"

    # Enhanced training command
    PYTHONPATH="$PYTHONPATH" bash ppo_optimized/launch_scripts/train_ppo_online.sh \
        $STEPS_PER_SCENARIO sac_transfer \
        --enable_enhanced_features $ENABLE_ENHANCED_FEATURES \
        --state_history_window $STATE_HISTORY_WINDOW \
        --qps_window $QPS_WINDOW \
        --latency_threshold $LATENCY_THRESHOLD \
        --latency_penalty_scale $LATENCY_PENALTY_SCALE \
        --load_balance_penalty $LOAD_BALANCE_PENALTY \
        --enable_dynamic_temperature $ENABLE_DYNAMIC_TEMPERATURE \
        --base_temperature $BASE_TEMPERATURE \
        --min_temperature $MIN_TEMPERATURE \
        --max_temperature $MAX_TEMPERATURE \
        --qps_sensitivity $QPS_SENSITIVITY \
        --latency_sensitivity $LATENCY_SENSITIVITY \
        --metrics_export_enabled $METRICS_EXPORT_ENABLED \
        --metrics_export_format csv \
        --metrics_export_path "$SCENARIO_DIR/exports" \
        --tensorboard_log_dir "$SCENARIO_DIR/tensorboard" \
        --checkpoint_dir "$SCENARIO_DIR/checkpoints" || {
            echo "❌ Training failed for QPS $qps"
            continue
        }

    echo "✅ QPS $qps training completed"

    # Check for exported metrics
    if [ -d "$SCENARIO_DIR/exports" ]; then
        export_files=$(find "$SCENARIO_DIR/exports" -name "*.csv" | wc -l)
        echo "📈 Exported $export_files CSV files"
    fi

    echo ""
done

echo "🎯 Enhanced PPO Multi-QPS Training Summary"
echo "========================================="

for qps in "${QPS_SCENARIOS[@]}"; do
    SCENARIO_DIR="outputs/runs/enhanced_ppo/qps_${qps}"

    if [ -d "$SCENARIO_DIR" ]; then
        echo "QPS $qps:"

        # Check TensorBoard logs
        if [ -d "$SCENARIO_DIR/tensorboard" ]; then
            tb_files=$(find "$SCENARIO_DIR/tensorboard" -name "events.out.tfevents.*" | wc -l)
            echo "  📊 TensorBoard: $tb_files event files"
        fi

        # Check exported metrics
        if [ -d "$SCENARIO_DIR/exports" ]; then
            csv_files=$(find "$SCENARIO_DIR/exports" -name "*.csv" | wc -l)
            echo "  📈 Exports: $csv_files CSV files"

            # Show latest CSV sample if available
            latest_csv=$(find "$SCENARIO_DIR/exports" -name "*.csv" -type f -exec ls -t {} + | head -1)
            if [ -n "$latest_csv" ]; then
                echo "  📄 Latest export: $(basename "$latest_csv")"
                if [ -f "$latest_csv" ]; then
                    lines=$(wc -l < "$latest_csv")
                    echo "  📝 Records: $lines lines"
                fi
            fi
        fi

        # Check checkpoints
        if [ -d "$SCENARIO_DIR/checkpoints" ]; then
            checkpoint_files=$(find "$SCENARIO_DIR/checkpoints" -name "*.pt" | wc -l)
            echo "  💾 Checkpoints: $checkpoint_files saved models"
        fi

        echo ""
    else
        echo "QPS $qps: ❌ No output directory found"
        echo ""
    fi
done

echo "🔍 Analysis Instructions:"
echo "========================"
echo "1. View TensorBoard:"
echo "   tensorboard --logdir outputs/runs/enhanced_ppo --port 6007"
echo ""
echo "2. Analyze CSV exports for:"
echo "   - reward_latency_penalty (should activate when latency > $LATENCY_THRESHOLD)"
echo "   - reward_load_balance_penalty (should penalize replica hot-spotting)"
echo "   - temperature (should decrease under high QPS/latency pressure)"
echo "   - temp_qps_pressure_ema, temp_latency_pressure_ema (pressure indicators)"
echo ""
echo "3. Compare scenarios:"
echo "   - QPS 1.5: Low pressure, higher exploration"
echo "   - QPS 2.0: Medium pressure, balanced exploration"
echo "   - QPS 2.5: High pressure, reduced exploration"
echo ""
echo "✅ Enhanced PPO Multi-QPS Training Complete!"