#!/usr/bin/env bash
# Start TensorBoard against reproducible run directories.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs/.tensorboard
setsid tensorboard --logdir ./runs --host 127.0.0.1 --port 6006 \
  > ./runs/.tensorboard/tensorboard.log 2>&1 < /dev/null &
echo $! > ./runs/.tensorboard/tensorboard.pid
echo "TensorBoard started (pid $(cat ./runs/.tensorboard/tensorboard.pid)) -> http://127.0.0.1:6006"
