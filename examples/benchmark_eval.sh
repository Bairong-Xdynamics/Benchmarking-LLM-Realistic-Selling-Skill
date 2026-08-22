#!/bin/bash

# Configuration
USER_API_ENDPOINT="http://YOUR_USER_API_ENDPOINT"
USER_API_KEY="YOUR_USER_API_KEY"
USER_MODEL_NAME="YOUR_USER_MODEL_NAME" 
ASSISTANT_API_ENDPOINT="https://YOUR_ASSISTANT_API_ENDPOINT"
ASSISTANT_API_KEY="YOUR_ASSISTANT_API_KEY"

MODEL="YOUR_ASSISTANT_MODEL_NAME"
EXTRA_BODY='{"reasoning": {"enabled": false}}'

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAX_SAMPLES=100 # Default to -1 if not provided as first argument

echo "----------------------------------------------------------------"
echo "Running evaluation for model: $MODEL"
echo "Extra Body: $EXTRA_BODY"
echo "Max Samples: $MAX_SAMPLES"

# Function to run evaluation
run_eval() {
    local LANGUAGE=$1
    local INPUT_FILE=$2
    local OUTPUT_DIR=$3

    # Create log file name
    local SAFE_MODEL_NAME=${MODEL//\//_}
    local LOG_FILE="logs/salesllm_eval_${SAFE_MODEL_NAME}_${LANGUAGE}_${TIMESTAMP}.log"
    mkdir -p logs

    echo "----------------------------------------------------------------"
    echo "Starting evaluation for language: $LANGUAGE"
    echo "Input File: $INPUT_FILE"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Log file: ${LOG_FILE}"

    nohup python3 salesllm/salesllm_evaluation.py \
      --assistant_model_name "$MODEL" \
      --user_model_name "$USER_MODEL_NAME" \
      --assistant_API_end_point "$ASSISTANT_API_ENDPOINT" \
      --assistant_API_key "$ASSISTANT_API_KEY" \
      --user_API_end_point "$USER_API_ENDPOINT" \
      --user_API_key "$USER_API_KEY" \
      --assistant_api_type "openai" \
      --user_api_type "other" \
      --user_api_version "" \
      --execution_mode "concurrent" \
      --max_workers "1" \
      --round_num "20" \
      --input_file "$INPUT_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --language "$LANGUAGE" \
      --save_intermediate \
      --max_samples "$MAX_SAMPLES" \
      --assistant_extra_body "$EXTRA_BODY" \
      > "$LOG_FILE" 2>&1 &
      
    local PID=$!
    echo "Process started with PID: $PID"
    wait $PID
    echo "Finished evaluation for $LANGUAGE"
}

# 1. Run Chinese (zh) evaluation
run_eval "zh" "./data/benchmark/conversations_1000_zh.jsonl" "./results/zh/"

# 2. Run English (en) evaluation
run_eval "en" "./data/benchmark/conversations_805_en.jsonl" "./results/en/"

echo "----------------------------------------------------------------"
echo "All evaluations completed."
