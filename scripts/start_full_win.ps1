param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [bool]$StartQGroundControl = $true
)

$ErrorActionPreference = "Stop"

function Test-PortOpen {
    param(
        [string]$Hostname,
        [int]$Port
    )
    $client = New-Object System.Net.Sockets.TcpClient
    try {
    $iar = $client.BeginConnect($Hostname, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(500)
        if (-not $ok) { return $false }
        $client.EndConnect($iar)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Start-QGroundControlApp {
    if (-not $StartQGroundControl) {
        Write-Host "[start_full_win] QGroundControl launch skipped (StartQGroundControl=false)."
        return
    }

    if (Get-Process -Name "QGroundControl" -ErrorAction SilentlyContinue) {
        Write-Host "[start_full_win] QGroundControl already running."
        return
    }

    $candidates = @(
        (Join-Path $env:ProgramFiles "QGroundControl\QGroundControl.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\QGroundControl\QGroundControl.exe")
    ) | Where-Object { $_ -and (Test-Path $_) }

    foreach ($exe in $candidates) {
        Start-Process -FilePath $exe | Out-Null
        Write-Host "[start_full_win] Started QGroundControl: $exe"
        return
    }

    try {
        Start-Process -FilePath "QGroundControl" | Out-Null
        Write-Host "[start_full_win] Started QGroundControl from PATH."
        return
    }
    catch {
        Write-Warning "[start_full_win] QGroundControl not found. Install it or add it to PATH."
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv-web\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[start_full_win] Creating .venv-web..."
    python -m venv .venv-web
    if (-not (Test-Path $venvPython)) {
        throw "Failed to create .venv-web (python -m venv .venv-web)."
    }
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r "backend\requirements.txt"
}

$npmCmd = "npm.cmd"
if (-not (Get-Command $npmCmd -ErrorAction SilentlyContinue)) {
    $fallbackNpm = "C:\Program Files\nodejs\npm.cmd"
    if (Test-Path $fallbackNpm) {
        $npmCmd = $fallbackNpm
    }
    else {
        throw "npm was not found. Install Node.js LTS first."
    }
}

$frontendDir = Join-Path $repoRoot "frontend"
$viteBin = Join-Path $frontendDir "node_modules\.bin\vite.cmd"
if (-not (Test-Path $viteBin)) {
    Write-Host "[start_full_win] Installing frontend dependencies..."
    & $npmCmd --prefix "$frontendDir" install
}

if (Test-PortOpen -Hostname "127.0.0.1" -Port $BackendPort) {
    Write-Host "[start_full_win] Backend already running on :$BackendPort"
}
else {
    Write-Host "[start_full_win] Starting backend on :$BackendPort"
    Start-Process -FilePath $venvPython -WorkingDirectory $repoRoot -ArgumentList "-m uvicorn backend.app.main:app --host 127.0.0.1 --port $BackendPort" | Out-Null
}

if (Test-PortOpen -Hostname "127.0.0.1" -Port $FrontendPort) {
    Write-Host "[start_full_win] Frontend already running on :$FrontendPort"
}
else {
    Write-Host "[start_full_win] Starting frontend on :$FrontendPort"
    $cmdExe = "$env:WINDIR\System32\cmd.exe"
    $cmdArgs = "/c","set PATH=C:\Program Files\nodejs;%PATH% && cd /d `"$frontendDir`" && npm run dev -- --host 127.0.0.1 --port $FrontendPort"
    Start-Process -FilePath $cmdExe -WorkingDirectory $repoRoot -ArgumentList $cmdArgs | Out-Null
}

Start-QGroundControlApp

$backendUp = $false
$frontendUp = $false
$deadline = (Get-Date).AddSeconds(20)
do {
    $backendUp = Test-PortOpen -Hostname "127.0.0.1" -Port $BackendPort
    $frontendUp = Test-PortOpen -Hostname "127.0.0.1" -Port $FrontendPort
    if ($backendUp -and $frontendUp) { break }
    Start-Sleep -Seconds 1
} while ((Get-Date) -lt $deadline)

Write-Host "[start_full_win] Backend listening: $backendUp (http://127.0.0.1:$BackendPort)"
Write-Host "[start_full_win] Frontend listening: $frontendUp (http://127.0.0.1:$FrontendPort)"
if (-not ($backendUp -and $frontendUp)) {
    throw "One or more services failed to start."
}
