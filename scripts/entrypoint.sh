#!/bin/bash
set -euo pipefail

CMD="${1:-train}"
shift || true

case "$CMD" in
  train|validate|infer)
    exec poker-yolo "$CMD" "$@"
    ;;
  mlflow)
    exec mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri "${MLFLOW_BACKEND_STORE_URI:-sqlite:////mlflow/mlflow.db}" \
      --default-artifact-root "${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts}"
    ;;
  bash|shell)
    exec /bin/bash "$@"
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Usage: entrypoint.sh {train|validate|infer|mlflow|bash}"
    exit 1
    ;;
esac
