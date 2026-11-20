# Daily paper-trading cycle, for Windows Task Scheduler.
#   dry run:  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\live\run_daily.ps1
#   live:     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\live\run_daily.ps1 -Live
# Run it on weekdays between 19:00 ET the evening before and ~09:00 ET (e.g. 13:30 Polish time).
# The summary of each run is appended to state\logs\run_daily.log; details go to state\logs\tool.log.
param([switch]$Live)

$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $repo
New-Item -ItemType Directory -Force "state\logs" | Out-Null
$log = Join-Path $repo "state\logs\run_daily.log"

$cliArgs = @("-m", "apps.cli", "run-daily")
if ($Live) { $cliArgs += "--live" }

"===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') run-daily $(if ($Live) {'LIVE'} else {'dry'}) =====" |
    Out-File -FilePath $log -Append -Encoding utf8
& "$repo\.venv\Scripts\python.exe" @cliArgs 2>&1 | ForEach-Object { "$_" } |
    Out-File -FilePath $log -Append -Encoding utf8
$code = $LASTEXITCODE
"exit code $code" | Out-File -FilePath $log -Append -Encoding utf8
exit $code
