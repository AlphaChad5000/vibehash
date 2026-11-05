# Install the Vibe Hash app on Windows and start mining, with no questions. In PowerShell:
#
#     irm https://vibehash.org/app.ps1 | iex
#
# The same thing VibeHash.exe does, without a download Windows SmartScreen warns about (a pasted line never trips it;
# an unsigned .exe, .msi or .bat does). It puts uv (one pinned release, checked against its published SHA-256) and
# the launcher (launcher/vibehash_launcher.py) in %LOCALAPPDATA%\VibeHash, adds "Vibe Hash" to the Start menu, and
# runs the launcher, which installs a Python of its own, Vibe Hash and the GPU build of PyTorch, then mines.
# Safe to run again. ASCII only: irm may not decode this file as UTF-8.

& {  # a script block, so a truncated download runs nothing and nothing here leaks into your PowerShell session
    $ErrorActionPreference = 'Stop'
    $uvVersion = '0.13.7'  # keep in step with UV_VERSION in launcher/vibehash_launcher.py (tests/test_launcher.py checks)
    $site = if ($env:VIBEHASH_SITE) { $env:VIBEHASH_SITE } else { 'https://vibehash.org' }
    $dir = if ($env:VIBEHASH_LAUNCHER_DIR) { $env:VIBEHASH_LAUNCHER_DIR } else { Join-Path $env:LOCALAPPDATA 'VibeHash' }
    New-Item -ItemType Directory -Force $dir | Out-Null
    Write-Host "`n  Vibe Hash  -  mining that trains real AI models  -  vibehash.org`n"

    $arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'aarch64' } else { 'x86_64' }
    $asset = "uv-$arch-pc-windows-msvc.zip"
    $uv = Join-Path $dir 'uv.exe'
    $stamp = Join-Path $dir 'uv.version'
    if (-not ((Test-Path $uv) -and (Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $uvVersion))) {
        Write-Host "  getting uv $uvVersion (the installer's installer, about 15 MB)..."
        $url = "https://github.com/astral-sh/uv/releases/download/$uvVersion/$asset"
        $zip = Join-Path $dir $asset
        $ProgressPreference = 'SilentlyContinue'  # Windows PowerShell's progress bar makes downloads crawl
        Invoke-WebRequest $url -OutFile $zip -UseBasicParsing
        $want = ((Invoke-WebRequest "$url.sha256" -UseBasicParsing).Content -split '\s+')[0]
        if ($want -is [byte[]]) { $want = [Text.Encoding]::ASCII.GetString($want) }
        if ((Get-FileHash $zip -Algorithm SHA256).Hash -ne $want.Trim().ToUpper()) {
            Remove-Item $zip
            Write-Host "  The download of $asset doesn't match the checksum its release publishes; not installing it."
            Write-Host "  Check your connection (a proxy?) and try again."
            return
        }
        $unpack = Join-Path $dir 'uv-unpack'
        Expand-Archive $zip $unpack -Force
        Copy-Item (Get-ChildItem $unpack -Recurse -Filter 'uv.exe' | Select-Object -First 1).FullName $uv -Force
        Remove-Item $zip, $unpack -Recurse -Force
        Set-Content $stamp $uvVersion -Encoding ascii
    }

    $launcher = Join-Path $dir 'vibehash_launcher.py'
    Invoke-WebRequest "$site/vibehash_launcher.py" -OutFile $launcher -UseBasicParsing

    # VibeHash.cmd runs the launcher on a Python uv fetches (the computer's own Python is never touched), with uv's
    # Python and cache kept in this folder as the launcher keeps its own (child_env), so --uninstall removes them all.
    $cmd = Join-Path $dir 'VibeHash.cmd'
    $lines = '@echo off',
        "set `"UV_PYTHON_INSTALL_DIR=$dir\python`"",
        "set `"UV_CACHE_DIR=$dir\cache`"",
        'set "UV_PYTHON_PREFERENCE=only-managed"',
        "`"$uv`" run --quiet --no-project --python 3.12 `"$launcher`" %*"
    Set-Content $cmd $lines -Encoding ascii

    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Programs')) 'Vibe Hash.lnk'))
    $link.TargetPath = $cmd
    $link.WorkingDirectory = $dir
    $link.Description = 'Vibe Hash: mining that trains real AI models'
    $link.Save()
    Write-Host "  Added Vibe Hash to the Start menu. Starting it now (Ctrl+C stops it)...`n"

    & $cmd
}
