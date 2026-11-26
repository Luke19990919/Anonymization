$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "需要先安裝 Git。" }
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "建立環境失敗；請確認已安裝 Python 3.11。" }
& .\.venv\Scripts\python.exe scripts/fetch_assets.py
if ($LASTEXITCODE -ne 0) { throw "字型資產核對失敗。" }
& .\.venv\Scripts\python.exe -m pip install -c constraints-tested.txt .
if ($LASTEXITCODE -ne 0) { throw "套件安裝失敗。" }
Write-Host "安裝完成。雙擊 啟動影片處理.bat 開啟處理頁。"
