#!/bin/bash

# ./kill_training.sh
echo "Finding running training processes..."

# Find training processes
TRAINING_PIDS=$(ps aux | grep -E "python.*train_lstm_hctsa|python.*train_" | grep -v grep | awk '{print $2}')

if [ -z "$TRAINING_PIDS" ]; then
    echo "No training processes found"
else
    echo "Killing training processes: $TRAINING_PIDS"
    kill -9 $TRAINING_PIDS
    sleep 2
    echo "Training processes killed"
fi

# Find any TensorFlow/Keras processes
TF_PIDS=$(ps aux | grep -E "python.*tensorflow|python.*keras" | grep -v grep | awk '{print $2}')

if [ -z "$TF_PIDS" ]; then
    echo "No TensorFlow processes found"
else
    echo "Killing TensorFlow processes: $TF_PIDS"
    kill -9 $TF_PIDS
    sleep 2
    echo "TensorFlow processes killed"
fi

echo "Current memory usage:"
top -l 1 | head -7 | tail -1

echo "Process cleanup complete!"
