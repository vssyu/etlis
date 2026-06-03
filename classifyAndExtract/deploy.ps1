# =============================================================================
# deploy.ps1  —  Copy project to Linux server and set up Docker environment
# =============================================================================
# Usage:
#   .\deploy.ps1
#   .\deploy.ps1 -RemoteUser ubuntu -RemoteHost 10.0.0.5
#
# You will be prompted for your SSH password twice:
#   1. scp  — uploads the archive
#   2. ssh  — extracts files and runs setup.sh (installs Docker + builds image)
#
# setup.sh requires root. If your user is not root, it will call sudo internally.
# =============================================================================

param(
    [string]$RemoteUser = "ubuntu",
    [string]$RemoteHost = "YOUR_SERVER_IP_OR_HOSTNAME",
    [int]   $RemotePort = 22,
    [string]$RemoteDir  = "/home/ubuntu/ClassfierAndExtractor"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Validate config ───────────────────────────────────────────────────────────
if ($RemoteHost -eq "YOUR_SERVER_IP_OR_HOSTNAME") {
    Write-Error "Set -RemoteHost <ip> or edit the default value at the top of this script."
    exit 1
}

$LocalDir  = $PSScriptRoot
$Archive   = "$env:TEMP\classifier_deploy.zip"
$RemoteZip = "/tmp/classifier_deploy.zip"

# ── 1. Package project files ──────────────────────────────────────────────────
Write-Host "`n[1/3] Packaging project files..."

$include  = @("*.py", "*.txt", "*.sh", "*.yml", "*.yaml", "Dockerfile", ".dockerignore")
$exclude  = @("output", "__pycache__", ".idea", ".git", "venv")

$filesToPack = Get-ChildItem -Path $LocalDir -File -Recurse | Where-Object {
    $rel = $_.FullName.Substring($LocalDir.Length + 1)
    $inExcluded = $false
    foreach ($ex in $exclude) {
        if ($rel -like "$ex*") { $inExcluded = $true; break }
    }
    if ($inExcluded) { return $false }
    foreach ($pat in $include) {
        if ($_.Name -like $pat) { return $true }
    }
    return $false
}

if ($filesToPack.Count -eq 0) {
    Write-Error "No files found to package. Run this script from the project directory."
    exit 1
}

if (Test-Path $Archive) { Remove-Item $Archive -Force }
Compress-Archive -Path $filesToPack.FullName -DestinationPath $Archive
Write-Host "    Packed $($filesToPack.Count) files -> $Archive"

# ── 2. Upload archive ─────────────────────────────────────────────────────────
Write-Host "`n[2/3] Uploading to ${RemoteUser}@${RemoteHost} ..."
Write-Host "      (SSH password prompt #1)"
scp -P $RemotePort $Archive "${RemoteUser}@${RemoteHost}:${RemoteZip}"

# ── 3. Remote: extract + install Docker + build image ────────────────────────
Write-Host "`n[3/3] Running remote setup (Docker install + image build)..."
Write-Host "      (SSH password prompt #2)"
Write-Host "      Note: Docker image build downloads ~5 GB on first run."

# Use sudo for Docker/toolkit installation; the project files are owned by the user.
$RemoteCmd = @"
set -e

# Extract project files as the logged-in user
mkdir -p '$RemoteDir'
cd '$RemoteDir'
unzip -o '$RemoteZip' -d .
rm '$RemoteZip'
chmod +x setup.sh

# Run setup.sh with sudo (needed for apt-get and Docker installation)
if [ "\$(id -u)" -eq 0 ]; then
    bash setup.sh
else
    sudo bash setup.sh
fi
"@

ssh -p $RemotePort "${RemoteUser}@${RemoteHost}" $RemoteCmd

# ── Cleanup ───────────────────────────────────────────────────────────────────
Remove-Item $Archive -Force

Write-Host ""
Write-Host "Deploy complete."
Write-Host "Project : $RemoteDir on $RemoteHost"
Write-Host ""
Write-Host "Next — upload your data files:"
Write-Host "  scp -P $RemotePort clauses.xlsx annotated.xlsx ``"
Write-Host "    ${RemoteUser}@${RemoteHost}:${RemoteDir}/data/"
Write-Host ""
Write-Host "Then open a shell inside the container:"
Write-Host "  ssh -p $RemotePort ${RemoteUser}@${RemoteHost}"
Write-Host "  cd $RemoteDir && docker compose run --rm classifier bash"
