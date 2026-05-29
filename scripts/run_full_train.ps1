# Full GPU train (default.yaml) + observability stack.

# Rebuild the image when poker_yolo code changes so report JSON matches the dashboard.

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..



docker compose build poker-yolo

docker compose up -d mlflow

docker compose --profile observability up -d



# ~80% of host RAM for the training container (matches gpu_resource_fraction in default.yaml)

$totalGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)

$memLimitGb = [math]::Max(4, [int][math]::Floor($totalGb * 0.8))

$env:POKER_YOLO_MEM_LIMIT = "${memLimitGb}g"

Write-Host "POKER_YOLO_MEM_LIMIT=$($env:POKER_YOLO_MEM_LIMIT) (host RAM ${totalGb} GB)"



docker compose run --rm -e REQUIRE_CUDA=1 poker-yolo --config configs/default.yaml train



$weights = "runs/classify/runs/train/poker_cards/weights/best.pt"

if (-not (Test-Path $weights)) {

    Write-Error "Weights not found: $weights"

}



docker compose --profile observability restart grafana report-server pushgateway



$mlflowPort = if ($env:MLFLOW_PORT) { $env:MLFLOW_PORT } else { "5000" }

$grafanaPort = if ($env:GRAFANA_PORT) { $env:GRAFANA_PORT } else { "3001" }

Write-Host "Done."

Write-Host "  MLflow:  http://localhost:${mlflowPort}"

Write-Host "  Grafana (training):  http://localhost:${grafanaPort}/d/poker-yolo-training/poker-yolo-training"
Write-Host "  Grafana (curves):    http://localhost:${grafanaPort}/d/poker-yolo-curves/poker-yolo-training-curves"
Write-Host "  Grafana (benchmark): http://localhost:${grafanaPort}/d/poker-yolo-inference/poker-yolo-benchmark-inference"


