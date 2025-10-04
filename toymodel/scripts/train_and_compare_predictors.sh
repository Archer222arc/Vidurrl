#!/bin/bash
###############################################################################
# Train and Compare Latency Predictors
#
# This script automates the complete workflow:
# 1. Train PPO with simple predictor
# 2. Train PPO with system-aware predictor
# 3. Compare performance of both models
###############################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SIMPLE_CONFIG="toymodel/configs/ppo_config_simple.json"
SYSTEM_AWARE_CONFIG="toymodel/configs/ppo_config_system_aware.json"
SIMPLE_OUTPUT_DIR="toymodel/outputs/simple_predictor"
SYSTEM_AWARE_OUTPUT_DIR="toymodel/outputs/system_aware_predictor"
COMPARISON_OUTPUT_DIR="toymodel/outputs/comparison"
NUM_EVAL_EPISODES=500

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     PPO Latency Predictor Training & Comparison Pipeline      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

###############################################################################
# Step 1: Train Simple Predictor
###############################################################################
echo -e "${YELLOW}[Step 1/3] Training Simple Predictor${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Config: $SIMPLE_CONFIG"
echo "Output: $SIMPLE_OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$SIMPLE_OUTPUT_DIR"

# Run training
echo -e "${GREEN}Starting training...${NC}"
python toymodel/scripts/train_ppo.py \
    --config "$SIMPLE_CONFIG" \
    --device cpu \
    2>&1 | tee "$SIMPLE_OUTPUT_DIR/training.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Simple predictor training completed${NC}"
else
    echo -e "${RED}✗ Simple predictor training failed${NC}"
    exit 1
fi
echo ""

###############################################################################
# Step 2: Train System-Aware Predictor
###############################################################################
echo -e "${YELLOW}[Step 2/3] Training System-Aware Predictor${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Config: $SYSTEM_AWARE_CONFIG"
echo "Output: $SYSTEM_AWARE_OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$SYSTEM_AWARE_OUTPUT_DIR"

# Run training
echo -e "${GREEN}Starting training...${NC}"
python toymodel/scripts/train_ppo.py \
    --config "$SYSTEM_AWARE_CONFIG" \
    --device cpu \
    2>&1 | tee "$SYSTEM_AWARE_OUTPUT_DIR/training.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ System-aware predictor training completed${NC}"
else
    echo -e "${RED}✗ System-aware predictor training failed${NC}"
    exit 1
fi
echo ""

###############################################################################
# Step 3: Compare Performance
###############################################################################
echo -e "${YELLOW}[Step 3/3] Comparing Predictor Performance${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Evaluation Episodes: $NUM_EVAL_EPISODES"
echo "Output: $COMPARISON_OUTPUT_DIR"
echo ""

# Create comparison output directory
mkdir -p "$COMPARISON_OUTPUT_DIR"

# Run comparison
echo -e "${GREEN}Running comparison...${NC}"
python toymodel/scripts/compare_predictors.py \
    --simple-config "$SIMPLE_CONFIG" \
    --simple-model "$SIMPLE_OUTPUT_DIR/models/ppo_model_latest.pt" \
    --system-aware-config "$SYSTEM_AWARE_CONFIG" \
    --system-aware-model "$SYSTEM_AWARE_OUTPUT_DIR/models/ppo_model_latest.pt" \
    --num-episodes "$NUM_EVAL_EPISODES" \
    --output-dir "$COMPARISON_OUTPUT_DIR" \
    2>&1 | tee "$COMPARISON_OUTPUT_DIR/comparison.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Performance comparison completed${NC}"
else
    echo -e "${RED}✗ Performance comparison failed${NC}"
    exit 1
fi
echo ""

###############################################################################
# Summary
###############################################################################
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      Pipeline Complete                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Results Summary:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Output Directories:"
echo "   Simple Predictor:      $SIMPLE_OUTPUT_DIR"
echo "   System-Aware Predictor: $SYSTEM_AWARE_OUTPUT_DIR"
echo "   Comparison Results:     $COMPARISON_OUTPUT_DIR"
echo ""
echo "📊 Key Files:"
echo "   Simple Model:           $SIMPLE_OUTPUT_DIR/models/ppo_model_latest.pt"
echo "   System-Aware Model:     $SYSTEM_AWARE_OUTPUT_DIR/models/ppo_model_latest.pt"
echo "   Comparison JSON:        $COMPARISON_OUTPUT_DIR/comparison_results.json"
echo "   Comparison Plot:        $COMPARISON_OUTPUT_DIR/predictor_comparison.png"
echo ""
echo "📈 TensorBoard:"
echo "   Simple Predictor:       tensorboard --logdir $SIMPLE_OUTPUT_DIR/tensorboard --port 6006"
echo "   System-Aware Predictor: tensorboard --logdir $SYSTEM_AWARE_OUTPUT_DIR/tensorboard --port 6007"
echo ""
echo -e "${GREEN}✓ All tasks completed successfully!${NC}"
echo ""
