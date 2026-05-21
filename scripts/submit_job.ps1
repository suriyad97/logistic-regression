#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Submit Azure ML jobs directly from local machine.
  Requires: az cli + ml extension, az login already done.

.USAGE
  .\scripts\submit_job.ps1 -Job train
  .\scripts\submit_job.ps1 -Job data-prep
  .\scripts\submit_job.ps1 -Job monitor
  .\scripts\submit_job.ps1 -Job inference
  .\scripts\submit_job.ps1 -Job env          # recreate environment
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("train","data-prep","monitor","inference","env")]
    [string]$Job,

    [string]$ResourceGroup = "ml-pipeline-rg",
    [string]$Workspace     = "logistic-regression-ws",
    [switch]$Tune,          # pass --tune flag to training job
    [switch]$Watch          # stream logs after submission
)

$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

function Submit-Job($YamlFile, $ExtraArgs = @()) {
    Write-Host "`n🚀 Submitting: $YamlFile" -ForegroundColor Cyan
    $cmd = @(
        "az", "ml", "job", "create",
        "--file", $YamlFile,
        "--resource-group", $ResourceGroup,
        "--workspace-name", $Workspace,
        "--output", "table"
    ) + $ExtraArgs

    $result = & $cmd[0] $cmd[1..($cmd.Length-1)]
    Write-Host $result

    # Extract job name for streaming
    $jobName = ($result | Select-String -Pattern "^\S+" | Select-Object -First 1).Line.Trim()
    if ($Watch -and $jobName) {
        Write-Host "`n📋 Streaming logs for job: $jobName" -ForegroundColor Yellow
        az ml job stream --name $jobName --resource-group $ResourceGroup --workspace-name $Workspace
    }
}

switch ($Job) {
    "env" {
        Write-Host "`n🏗  Creating/updating AML environment..." -ForegroundColor Green
        az ml environment create `
            --file config/environment.yml `
            --resource-group $ResourceGroup `
            --workspace-name $Workspace `
            --output table
    }

    "data-prep" {
        Submit-Job "mlpipelines/data_preparation/data_prep_job.yml"
    }

    "train" {
        $extraArgs = @()
        if ($Tune) {
            Write-Host "⚙️  Optuna tuning ENABLED (50 trials)" -ForegroundColor Magenta
            $extraArgs += @("--set", "environment_variables.ENABLE_TUNING=true")
            $extraArgs += @("--set", "environment_variables.N_TRIALS=50")
        }
        Submit-Job "mlpipelines/training/training_job.yml" $extraArgs
    }

    "monitor" {
        Submit-Job "mlpipelines/monitoring/monitoring_job.yml"
    }

    "inference" {
        Submit-Job "mlpipelines/inference/inference_job.yml"
    }
}
