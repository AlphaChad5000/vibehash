# Install Vibe Hash on Windows, then start its guided setup. In PowerShell:
#
#     irm https://vibehash.org/install.ps1 | iex
#
# Finds Python 3.10 or newer (and offers to install it with winget if there isn't one), installs or updates
# Vibe Hash from PyPI, then runs `python -m vibehash setup`, which asks the rest. Safe to run again.
# To join a network other than the public one, set its node first: $env:VIBEHASH_NODE = "192.168.1.20"

& {  # a script block, so a truncated download runs nothing and nothing here leaks into your PowerShell session
    function Find-Python {
        # `python` first, the name the guide's commands use; then the py launcher.
        foreach ($cmd in 'python', 'py -3') {
            $exe, $rest = $cmd -split ' '
            if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
            # Windows' "python" shortcut to the Microsoft Store fails this check, as does a Python older than 3.10
            $ok = & $exe @($rest) -c "import sys; print(sys.version_info >= (3, 10))" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ok -eq 'True') { return $cmd }
        }
    }

    function Show-Banner {
        # The mining dashboard's lettering and gold-to-violet gradient. Spelt in ASCII (# full block, ^ upper half,
        # _ lower half) because irm may not decode this file as UTF-8.
        $rows = '# # # #__ #^^   # # _^# #^ # #', '^_^ # #_# ##_   #^# #^# _# #^#'
        $tails = 'useful-work miner', "installer $([char]0xB7) vibehash.org"
        $grad = '255;205;60', '255;150;70', '255;95;150', '180;110;255'
        $color = -not ($env:NO_COLOR -or [Console]::IsOutputRedirected)
        $e = [char]27
        Write-Host ''
        for ($i = 0; $i -lt 2; $i++) {
            $row = $rows[$i].Replace('#', [char]0x2588).Replace('^', [char]0x2580).Replace('_', [char]0x2584)
            $line = ' '
            for ($j = 0; $j -lt $row.Length; $j++) {
                if ($color) { $line += "$e[1;38;2;$($grad[[int][math]::Floor($j * 4 / $row.Length)])m" }
                $line += $row[$j]
            }
            if ($color) { $line += "$e[0m   $e[38;2;125;135;148m$($tails[$i])$e[0m" } else { $line += "   $($tails[$i])" }
            Write-Host $line
        }
        Write-Host ''
    }

    Show-Banner
    $python = Find-Python
    if (-not $python -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "Vibe Hash runs on Python 3.10 or newer, which this computer doesn't have yet."
        $answer = Read-Host "Install Python 3.14 from python.org now (about 30 MB, with winget)? [Y/n]"
        if ($answer -match '^\s*(y|yes)?\s*$') {
            # just for you, with python.exe on your PATH
            winget install --exact --id Python.Python.3.14 --override "/passive InstallAllUsers=0 PrependPath=1 Include_launcher=1"
            # winget changes PATH for new windows only, so read it again for this one
            $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
            $python = Find-Python
        }
    }
    if (-not $python) {
        Write-Host "Vibe Hash needs Python 3.10 or newer. Install it from https://www.python.org/downloads/"
        Write-Host "(tick `"Add python.exe to PATH`"), open a new PowerShell window, and run this again."
        return
    }

    $exe, $rest = $python -split ' '
    Write-Host "Installing Vibe Hash with $python..."
    & $exe @($rest) -m pip install --upgrade --quiet --disable-pip-version-check vibehash
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nInstalling didn't work; pip's message above says why. Check you're online, then run this again."
        return
    }
    & $exe @($rest) -c "from importlib.metadata import version; print('Vibe Hash', version('vibehash'), 'is installed.')"
    Write-Host ""
    & $exe @($rest) -m vibehash setup
    if ($exe -ne 'python') {
        Write-Host "`nOn this computer Python is called py, so where the guide says ``python -m vibehash``, type ``py -m vibehash``."
    }
}
