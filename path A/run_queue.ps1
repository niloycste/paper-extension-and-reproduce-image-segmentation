param(
    [string]$Python = "python",
    [ValidateSet("cpu", "cuda", "auto")]
    [string]$Device = "cpu",
    [int]$Seed = 42,
    [switch]$SkipFinalize
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

function Invoke-Train {
    param(
        [string]$Name,
        [string[]]$TrainArgs
    )

    Write-Output "=========================================================="
    Write-Output "START $Name   [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]"
    Write-Output "=========================================================="
    & $Python -W ignore "path A\05_train.py" @TrainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped at $Name with exit code $LASTEXITCODE"
    }
    Write-Output "DONE  $Name   [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]"
}

function Get-LatestBestCheckpoint {
    param([string]$RunNamePattern)

    $runRoot = Join-Path "path A" "runs"
    $runDir = Get-ChildItem -LiteralPath $runRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like $RunNamePattern } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $runDir) {
        throw "No run directory matched pattern: $RunNamePattern"
    }

    $checkpoint = Join-Path $runDir.FullName "best.pth"
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        throw "Missing checkpoint: $checkpoint"
    }
    return $checkpoint
}

$common = @(
    "--dataset", "ClinicDB",
    "--runs", "1",
    "--seed", "$Seed",
    "--device", $Device
)

# Stage A is required for sparse top-1 warm start. It is included here so a
# fresh clone can reproduce the extension without manually locating a checkpoint.
$stageA = @(
    "--routing_mode", "adaptive_soft",
    "--kernel_sizes", "1", "3", "5",
    "--epoch", "200",
    "--lr", "0.0005",
    "--aux_supervision", "False"
)
Invoke-Train "stage A adaptive soft [1,3,5] @200ep" ($stageA + $common)

$stageABest = Get-LatestBestCheckpoint `
    "ClinicDB_MK_UNet_T_adaptive_soft_k135_auxFalse_e200_seed$Seed*"

$jobs = @(
    @{
        n = "control fixed [1,3,5] @100ep"
        a = @("--routing_mode", "fixed", "--kernel_sizes", "1", "3", "5",
              "--epoch", "100", "--lr", "0.0005", "--aux_supervision", "False")
    },
    @{
        n = "pruned fixed [1,3] @100ep"
        a = @("--routing_mode", "fixed", "--kernel_sizes", "1", "3",
              "--epoch", "100", "--lr", "0.0005", "--aux_supervision", "False")
    },
    @{
        n = "pruned fixed [1,5] @100ep"
        a = @("--routing_mode", "fixed", "--kernel_sizes", "1", "5",
              "--epoch", "100", "--lr", "0.0005", "--aux_supervision", "False")
    },
    @{
        n = "adaptive soft + auxiliary heads @100ep"
        a = @("--routing_mode", "adaptive_soft", "--epoch", "100",
              "--lr", "0.0005", "--aux_supervision", "True")
    },
    @{
        n = "predictive entropy + auxiliary heads @100ep"
        a = @("--routing_mode", "uncertainty_soft", "--epoch", "100",
              "--lr", "0.0005", "--aux_supervision", "True")
    },
    @{
        n = "pruned fixed [3,5] @100ep"
        a = @("--routing_mode", "fixed", "--kernel_sizes", "3", "5",
              "--epoch", "100", "--lr", "0.0005", "--aux_supervision", "False")
    },
    @{
        n = "fixed + auxiliary heads @100ep"
        a = @("--routing_mode", "fixed", "--epoch", "100",
              "--lr", "0.0005", "--aux_supervision", "True")
    },
    @{
        n = "sparse top-1 warm-started from stage A @60ep"
        a = @("--routing_mode", "sparse_top1", "--init_from", $stageABest,
              "--epoch", "60", "--lr", "0.0003", "--anneal_epochs", "20",
              "--aux_supervision", "False")
    }
)

foreach ($job in $jobs) {
    Invoke-Train $job.n ($job.a + $common)
}

if (-not $SkipFinalize) {
    & $Python -W ignore "path A\11_finalize.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Finalisation failed with exit code $LASTEXITCODE"
    }
}

Write-Output "Queue complete."
