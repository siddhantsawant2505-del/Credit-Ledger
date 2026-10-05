# Sequential runner for the remaining full-data retrain steps.
# Launched detached; progress is visible in models\sweep_*.log and models\oof_cache\.
$ErrorActionPreference = "Continue"
Set-Location "E:\Projects\Credit-Ledger"

Write-Output "=== [1/4] GB fold 5 ==="
python server/train_models.py --models gb --fold-start 5 --fold-end 5 *> models\sweep_gb5.log
Write-Output "=== [2/4] GB final fit (cached folds) ==="
python server/train_models.py --models gb --fold-end 5 *> models\sweep_gb_final.log
Write-Output "=== [3/4] DNN (all folds + final fit) ==="
python server/train_models.py --models dnn --fold-end 5 *> models\sweep_dnn.log
Write-Output "=== [4/4] Stacking rebuild from OOF cache ==="
python server/train_models.py --build-stack-from-cache *> models\sweep_stack.log
Write-Output "=== SWEEP COMPLETE ==="
