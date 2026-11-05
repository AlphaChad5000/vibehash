# Keeps this PC serving the site: runs the dispatcher (agents/dispatcher.py) and, if it ever stops (a crash, an error
# it can't get past), starts it again after a short pause. While this window is open the PC won't go to sleep, so a
# request or a chat message isn't left waiting; the power settings themselves aren't changed. Ctrl+C stops it all.
#
#   powershell -ExecutionPolicy Bypass -File agents/dispatch.ps1
#
# Needs what the dispatcher needs: .env.local with the API key and DISPATCH_TOKEN, and the training packages.

$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo  # agent-jobs/ and .env.local are at the repository's top

# The repo's .venv has a CPU-only PyTorch (fine for tests, a crawl for training), so use the first Python on PATH
# that isn't in it, even when this is started from inside the venv.
$venv = Join-Path $repo ".venv"
$python = Get-Command python -All -CommandType Application -ErrorAction SilentlyContinue |
    Where-Object { -not $_.Source.StartsWith($venv, [StringComparison]::OrdinalIgnoreCase) } |
    Select-Object -First 1 -ExpandProperty Source
if (-not $python) { $python = "python" }
Write-Host "using $python"

Add-Type -Namespace Muse -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[Muse.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null  # awake until this exits

try {
    while ($true) {
        Write-Host "$(Get-Date -Format 'HH:mm:ss') starting the dispatcher"
        & $python agents/dispatcher.py
        if ($LASTEXITCODE -eq 3) { break }  # no GPU (dispatcher.NO_GPU): starting it again won't help
        Write-Host "$(Get-Date -Format 'HH:mm:ss') the dispatcher stopped (exit code $LASTEXITCODE); starting it again in 15 s"
        Start-Sleep -Seconds 15
    }
} finally {
    [Muse.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null  # sleep as usual again
}
