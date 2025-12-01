#!/bin/bash

# -------- SETTINGS (EDIT ME) --------
PROJECT_DIR="$HOME/Documents/remote"
VENV_PATH="$HOME/venvs/gaec"
SCRIPT_PATH="$PROJECT_DIR/src/optuna_optimize.py"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/pids"
SIZES=(50 250 500 750)
# ------------------------------------

# Create log and pid directories if they don't exist
mkdir -p "$LOG_DIR" "$PID_DIR"

echo "[1] Activating virtual environment..."
source "$VENV_PATH/bin/activate"

for SIZE in "${SIZES[@]}"; do
    LOG_FILE="$LOG_DIR/optuna_$SIZE.log"
    PID_FILE="$PID_DIR/optuna_$SIZE.pid"

    echo "[2] Starting Optuna for size $SIZE with nohup..."
    nohup python "$SCRIPT_PATH" "$SIZE" >> "$LOG_FILE" 2>&1 &

    PID=$!
    echo $PID > "$PID_FILE"

    echo "---------------------------------------"
    echo " Optuna job launched for size $SIZE!"
    echo " PID:       $PID"
    echo " Log file:  $LOG_FILE"
    echo " PID file:  $PID_FILE"
    echo "---------------------------------------"
done

echo ""
echo "All jobs launched. You can safely close the terminal."
echo ""
echo "Monitor live output for a specific size:"
echo "    tail -f $LOG_DIR/optuna_<SIZE>.log"
echo ""
echo "Stop a specific job:"
echo "    kill \$(cat $PID_DIR/optuna_<SIZE>.pid)"
echo "---------------------------------------"
