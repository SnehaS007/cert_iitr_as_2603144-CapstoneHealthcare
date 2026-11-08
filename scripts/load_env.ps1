<#
.SYNOPSIS
    Load KEY=VALUE pairs from the project's .env into the current PowerShell session.

.DESCRIPTION
    PowerShell equivalent of `set -a && source .env && set +a`.
    Environment variables are process-wide, so a plain call (no dot-sourcing) is enough:
        .\scripts\load_env.ps1
    Variables last for the current terminal only; run again in each new terminal.

    Fails loudly on a missing file or a malformed line. Nothing is silently skipped
    except blank lines and lines starting with '#'.

.PARAMETER Path
    Path to the env file. Default: .env in the repository root (resolved from this
    script's location, so it works from any working directory).

.EXAMPLE
    Copy-Item .env.example .env      # once
    .\scripts\load_env.ps1
#>
[CmdletBinding()]
param(
    [string]$Path = (Join-Path (Split-Path -Parent $PSScriptRoot) ".env")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "Env file not found: $Path. Create it first: Copy-Item .env.example .env"
}

$lineNumber = 0
$loaded = [ordered]@{}
foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
    $lineNumber++
    $trimmed = $line.Trim()
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }

    if ($trimmed -notmatch '^([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$') {
        throw "Malformed line $lineNumber in ${Path}: '$line' (expected KEY=VALUE)"
    }
    $name = $Matches[1]
    $value = $Matches[2].Trim()
    if ($value.Length -ge 2 -and (($value[0] -eq '"' -and $value[-1] -eq '"') -or ($value[0] -eq "'" -and $value[-1] -eq "'"))) {
        $value = $value.Substring(1, $value.Length - 2)
    }

    Set-Item -LiteralPath "Env:$name" -Value $value
    $loaded[$name] = $value
}

if ($loaded.Count -eq 0) {
    throw "No variables found in $Path"
}

Write-Host "Loaded $($loaded.Count) variable(s) from ${Path}:"
foreach ($entry in $loaded.GetEnumerator()) {
    Write-Host ("  {0}={1}" -f $entry.Key, $entry.Value)
}
