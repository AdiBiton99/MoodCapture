# run_full_training.ps1
# Runs all training steps in order and shows a popup when done.
#
# Steps:
#   1. Build FER2013 train features     (skip if exists)
#   2. Build FER2013 test features      (skip if exists)
#   3. Build RAF-DB  train features     (skip if exists)
#   4. Build RAF-DB  test features      (skip if exists)
#   5. Train Fusion model (FER2013 + RAF-DB combined, class weights, larger MLP)
#
# Run:
#   powershell -ExecutionPolicy Bypass -File ml\run_full_training.ps1

$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot

$python = ".\.venv\Scripts\python.exe"
$env:PYTHONUNBUFFERED     = "1"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:GLOG_minloglevel     = "3"

function Show-Popup($message) {
    [System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms") | Out-Null
    [System.Windows.Forms.MessageBox]::Show(
        $message, "MoodCapture - Training",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null
}

# ── Step 1: FER2013 train ─────────────────────────────────────────────────────
if (Test-Path "data\fusion_features_train.npy") {
    Write-Host "[DONE]  Step 1/5: FER2013 train features already exist -- skipping." -ForegroundColor Green
} else {
    Write-Host "[START] Step 1/5: Building FER2013 TRAIN features..." -ForegroundColor Cyan
    & $python -u ml/build_fusion_dataset.py --split train 2>$null
    if ($LASTEXITCODE -ne 0) { Show-Popup "ERROR in Step 1"; exit 1 }
    Write-Host "[DONE]  Step 1/5: FER2013 train features ready." -ForegroundColor Green
}

# ── Step 2: FER2013 test ──────────────────────────────────────────────────────
if (Test-Path "data\fusion_features_test.npy") {
    Write-Host "[DONE]  Step 2/5: FER2013 test features already exist -- skipping." -ForegroundColor Green
} else {
    Write-Host "[START] Step 2/5: Building FER2013 TEST features..." -ForegroundColor Cyan
    & $python -u ml/build_fusion_dataset.py --split test 2>$null
    if ($LASTEXITCODE -ne 0) { Show-Popup "ERROR in Step 2"; exit 1 }
    Write-Host "[DONE]  Step 2/5: FER2013 test features ready." -ForegroundColor Green
}

# ── Step 3: RAF-DB train ──────────────────────────────────────────────────────
if (Test-Path "data\rafdb_features_train.npy") {
    Write-Host "[DONE]  Step 3/5: RAF-DB train features already exist -- skipping." -ForegroundColor Green
} else {
    Write-Host "[START] Step 3/5: Building RAF-DB TRAIN features (~1 hour)..." -ForegroundColor Cyan
    & $python -u ml/build_rafdb_dataset.py --split train 2>$null
    if ($LASTEXITCODE -ne 0) { Show-Popup "ERROR in Step 3"; exit 1 }
    Write-Host "[DONE]  Step 3/5: RAF-DB train features ready." -ForegroundColor Green
}

# ── Step 4: RAF-DB test ───────────────────────────────────────────────────────
if (Test-Path "data\rafdb_features_test.npy") {
    Write-Host "[DONE]  Step 4/5: RAF-DB test features already exist -- skipping." -ForegroundColor Green
} else {
    Write-Host "[START] Step 4/5: Building RAF-DB TEST features..." -ForegroundColor Cyan
    & $python -u ml/build_rafdb_dataset.py --split test 2>$null
    if ($LASTEXITCODE -ne 0) { Show-Popup "ERROR in Step 4"; exit 1 }
    Write-Host "[DONE]  Step 4/5: RAF-DB test features ready." -ForegroundColor Green
}

# ── Step 5: Train model ───────────────────────────────────────────────────────
Write-Host "[START] Step 5/5: Training Fusion model (FER2013 + RAF-DB, class weights)..." -ForegroundColor Cyan
& $python -u ml/train_fusion_model.py 2>$null
if ($LASTEXITCODE -ne 0) { Show-Popup "ERROR in Step 5 (training)"; exit 1 }
Write-Host "[DONE]  Step 5/5: Model trained and saved." -ForegroundColor Green

# ── Done ──────────────────────────────────────────────────────────────────────
[System.Media.SystemSounds]::Beep.Play()
Show-Popup "Training complete!`n`nModel saved: models\fusion_model.pkl`n`nRun the app: python main.py"
