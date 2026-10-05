# Finalize the 250-feature sweep: DNN final fit (folds cached) + stack rebuild.
$ErrorActionPreference = "Continue"
Set-Location "E:\Projects\Credit-Ledger"
Write-Output "=== [1/2] DNN final fit (cached OOF folds) ==="
python server/train_models.py --models dnn --fold-end 5 *> models\sweep_dnn_final.log
Write-Output "=== [2/2] Stacking rebuild from OOF cache ==="
python server/train_models.py --build-stack-from-cache *> models\sweep_stack_final.log
Write-Output "=== FINALIZE COMPLETE ==="
