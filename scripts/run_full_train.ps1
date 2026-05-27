$env:TEMP = "D:\cursor_univ\CV_proj\.tmp"
$env:TMP = "D:\cursor_univ\CV_proj\.tmp"
$env:MLFLOW_TRACKING_URI = "file:///D:/cursor_univ/CV_proj/mlruns"
$env:PROMETHEUS_PUSHGATEWAY_URL = ""
Set-Location "D:\cursor_univ\CV_proj"
Remove-Item "D:\cursor_univ\CV_proj\runs\train_full.log" -ErrorAction SilentlyContinue
& uv run poker-yolo --config configs/default.yaml train *>> "D:\cursor_univ\CV_proj\runs\train_full.log"
