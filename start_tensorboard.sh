#!/usr/bin/env bash
# Start TensorBoard against this repo's training logs.
set -euo pipefail
cd "$(dirname "$0")"
setsid tensorboard --logdir ./logs --host 127.0.0.1 --port 6006 \
  > ./tensorboard.log 2>&1 < /dev/null &
echo $! > ./tensorboard.pid
echo "TensorBoard started (pid $(cat ./tensorboard.pid)) -> http://127.0.0.1:6006"
