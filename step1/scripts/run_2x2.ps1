# 2x2 factorial: MK_UNet_T / ClinicDB / 352x352 / 200 epochs / seed 42, run SEQUENTIALLY.
#
#   A  baseline    -deep_supervision False  -ca_min_squeeze 1
#   B  +DS         -deep_supervision True   -ca_min_squeeze 1
#   C  +CA         -deep_supervision False  -ca_min_squeeze 4
#   D  +DS +CA     -deep_supervision True   -ca_min_squeeze 4
#
# Sequential on purpose. Each arm peaks near 7 GB, and two concurrent jobs will OOM
# on a 32 GB machine that is also running anything else. Sequential execution also
# keeps the wall-clock timings usable as reported efficiency numbers.
#
# Usage:   powershell -ExecutionPolicy Bypass -File scripts\run_2x2.ps1
# Resume:  see the --resume flag; each arm writes <run_id>-resume.pth every epoch.

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

# Point this at the python of your mkunetenv on THIS machine.
$PY = if ($env:MKUNET_PY) { $env:MKUNET_PY } else { "$env:USERPROFILE\.conda\envs\mkunetenv\python.exe" }
if (-not (Test-Path $PY)) { Write-Error "python not found at $PY - set `$env:MKUNET_PY"; exit 1 }

$OUT = 'experiments\results'
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

$common = @('--network','MK_UNet_T','--dataset','ClinicDB','--device','cpu',
            '--epoch','200','--runs','1','--seed','42')

$arms = @(
    @{ tag='A_baseline'; args=@('--deep_supervision','False','--ca_min_squeeze','1') },
    @{ tag='B_ds';       args=@('--deep_supervision','True', '--ca_min_squeeze','1') },
    @{ tag='C_ca';       args=@('--deep_supervision','False','--ca_min_squeeze','4') },
    @{ tag='D_ds_ca';    args=@('--deep_supervision','True', '--ca_min_squeeze','4') }
)

foreach ($arm in $arms) {
    $log = Join-Path $OUT ("run_" + $arm.tag + ".log")
    Write-Host "=== [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] starting arm $($arm.tag) -> $log"
    & $PY -W ignore train_polyp.py @common @($arm.args) *> $log
    Write-Host "=== [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] arm $($arm.tag) exited with $LASTEXITCODE"
}

Write-Host "=== [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] all arms complete"
