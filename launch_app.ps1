$ErrorActionPreference = "Stop"

$projectDir = $PSScriptRoot
$healthUrl = "http://127.0.0.1:8010/healthz"
$appUrl = "http://127.0.0.1:8010/"
$logDir = Join-Path $projectDir "storage"
$errorLog = Join-Path $logDir "launcher-error.log"
$serverOutputLog = Join-Path $logDir "server-output.log"
$serverErrorLog = Join-Path $logDir "server-error.log"
$serverScript = Join-Path $projectDir "run_server.py"
$startupTimeoutSeconds = 60

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Test-CopilotService {
    try {
        $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2
        return $response.status -eq "ok"
    }
    catch { return $false }
}

function Find-PythonExecutable {
    $candidates = @(
        (Join-Path $projectDir ".venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { return $pythonCommand.Source }
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonLauncher) { return $pythonLauncher.Source }
    throw "没有找到Python。请先按照README创建虚拟环境并安装requirements.txt。"
}

try {
    if (-not (Test-CopilotService)) {
        $pythonExe = Find-PythonExecutable
        Set-Content -LiteralPath $serverOutputLog -Value "" -Encoding UTF8
        Set-Content -LiteralPath $serverErrorLog -Value "" -Encoding UTF8
        $process = Start-Process `
            -FilePath $pythonExe `
            -ArgumentList ('"{0}"' -f $serverScript) `
            -WorkingDirectory $projectDir `
            -WindowStyle Hidden `
            -RedirectStandardOutput $serverOutputLog `
            -RedirectStandardError $serverErrorLog `
            -PassThru
        $maximumAttempts = $startupTimeoutSeconds * 2
        for ($attempt = 1; $attempt -le $maximumAttempts; $attempt++) {
            Start-Sleep -Milliseconds 500
            if (Test-CopilotService) { break }
            if ($process.HasExited) {
                $details = (Get-Content -LiteralPath $serverErrorLog -Raw -ErrorAction SilentlyContinue).Trim()
                if (-not $details) { $details = (Get-Content -LiteralPath $serverOutputLog -Raw -ErrorAction SilentlyContinue).Trim() }
                throw "服务启动失败。`n`n$details"
            }
        }
        if (-not (Test-CopilotService)) {
            throw "启动超过60秒，请检查8010端口或storage中的错误日志。"
        }
    }
    Start-Process $appUrl
}
catch {
    $message = $_.Exception.Message
    $message | Set-Content -LiteralPath $errorLog -Encoding UTF8
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        $message,
        "IncidentCopilot 启动失败",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    ) | Out-Null
}
