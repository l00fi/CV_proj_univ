#!/bin/bash
set -euo pipefail

# GPU if available, else CPU (see poker_yolo/device.py). Sets POKER_YOLO_DEVICE.
resolve_training_device() {
  local log
  log="$(mktemp)"
  if ! POKER_YOLO_DEVICE="$(python /app/scripts/resolve_device.py 2>"$log")"; then
    cat "$log" >&2
    rm -f "$log"
    exit 1
  fi
  cat "$log" >&2
  rm -f "$log"
  export POKER_YOLO_DEVICE
}

ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=("train")
fi

CMD="${ARGS[0]}"

if [[ "$CMD" == -* ]]; then
  CMD=""
  for arg in "${ARGS[@]}"; do
    case "$arg" in
      train|validate|infer)
        CMD="$arg"
        break
        ;;
    esac
  done
fi

case "$CMD" in
  train|validate|infer)
    resolve_training_device
    exec poker-yolo "${ARGS[@]}"
    ;;
  check-gpu|check-device)
    exec python /app/scripts/resolve_device.py --require-gpu
    ;;
  resolve-device)
    exec python /app/scripts/resolve_device.py
    ;;
  mlflow)
    exec mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri "${MLFLOW_BACKEND_STORE_URI:-sqlite:////mlflow/mlflow.db}" \
      --default-artifact-root "${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts}"
    ;;
  bash|shell)
    shift || true
    exec /bin/bash "$@"
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Usage: entrypoint.sh {train|validate|infer|check-gpu|resolve-device|mlflow|bash} [args...]"
    exit 1
    ;;
esac
