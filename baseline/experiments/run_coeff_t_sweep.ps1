$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".conda\python.exe"
$HYPIRRoot = Join-Path $ProjectRoot "HYPIR"
$InputDir = Join-Path $ProjectRoot "baseline\input"
$BaseModel = Join-Path $HYPIRRoot "models\stable-diffusion-2-1-base"
$WeightPath = Join-Path $HYPIRRoot "weights\HYPIR_sd2.pth"
$Candidates = @(200, 150, 100, 75, 50)

if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }
if (-not (Test-Path -LiteralPath $InputDir)) { throw "Input directory not found: $InputDir" }
if (-not (Test-Path -LiteralPath $BaseModel)) { throw "Base model not found: $BaseModel" }
if (-not (Test-Path -LiteralPath $WeightPath)) { throw "LoRA weight not found: $WeightPath" }

foreach ($CoeffT in $Candidates) {
    $ExperimentDir = Join-Path $ProjectRoot "baseline\experiments\coeff_t_$CoeffT"
    if ((Test-Path -LiteralPath $ExperimentDir) -and (Test-Path -LiteralPath (Join-Path $ExperimentDir "experiment_metadata.md"))) {
        throw "Refusing to overwrite existing experiment directory: $ExperimentDir"
    }
    $OutputDir = Join-Path $ExperimentDir "output"
    $ComparisonDir = Join-Path $ExperimentDir "comparison"
    New-Item -ItemType Directory -Force -Path $OutputDir, $ComparisonDir | Out-Null
    $LogPath = Join-Path $ExperimentDir "inference.log"
    $MetadataPath = Join-Path $ExperimentDir "experiment_metadata.md"
    $Arguments = @(
        "test.py",
        "--base_model_type", "sd2",
        "--base_model_path", "models\stable-diffusion-2-1-base",
        "--model_t", "200",
        "--coeff_t", "$CoeffT",
        "--lora_rank", "256",
        "--lora_modules", "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
        "--weight_path", "weights\HYPIR_sd2.pth",
        "--patch_size", "512",
        "--stride", "256",
        "--lq_dir", "..\baseline\input",
        "--scale_by", "factor",
        "--upscale", "1",
        "--captioner", "empty",
        "--output_dir", ("..\baseline\experiments\coeff_t_$CoeffT\output"),
        "--seed", "231",
        "--device", "cuda"
    )
    $CommandText = "& `"$Python`" " + (($Arguments | ForEach-Object { if ($_ -match "[ ,\\]") { "`"$_`"" } else { $_ } }) -join " ")
    $Start = Get-Date
    "[$Start] Starting coeff_t=$CoeffT" | Tee-Object -FilePath $LogPath
    Push-Location $HYPIRRoot
    try {
        # Redirect native stdout/stderr together; HYPIR progress output on stderr
        # must not be treated as a terminating PowerShell exception.
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python @Arguments *> $LogPath
        $ProcessExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousErrorActionPreference
        if ($ProcessExitCode -ne 0) { throw "HYPIR test.py failed for coeff_t=$CoeffT (exit $ProcessExitCode)" }
    }
    finally {
        Pop-Location
    }
    $End = Get-Date
    $TotalSeconds = [math]::Round(($End - $Start).TotalSeconds, 3)
    $LogText = Get-Content -LiteralPath $LogPath -Raw
    $LoadSeconds = ""
    $LoadMatch = [regex]::Match($LogText, "Models loaded in ([0-9.]+) seconds")
    if ($LoadMatch.Success) { $LoadSeconds = $LoadMatch.Groups[1].Value }
    @(
        "# coeff_t=$CoeffT experiment metadata",
        "",
        "- status: completed",
        "- total_elapsed_seconds: $TotalSeconds",
        "- model_load_seconds: $LoadSeconds",
        "- base_model_type: sd2",
        "- base_model_path: HYPIR/models/stable-diffusion-2-1-base",
        "- base_model_repo_id: sd-research/stable-diffusion-2-1-base",
        "- model_t: 200",
        "- coeff_t: $CoeffT",
        "- lora_rank: 256",
        "- lora_modules: to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
        "- patch_size: 512",
        "- stride: 256",
        "- scale_by: factor",
        "- upscale: 1",
        "- captioner: empty",
        "- seed: 231",
        "- device: cuda",
        "- input_dir: baseline/input",
        "- output_dir: baseline/experiments/coeff_t_$CoeffT/output",
        "- comparison_dir: baseline/experiments/coeff_t_$CoeffT/comparison",
        "- metrics_csv: baseline/experiments/coeff_t_$CoeffT/evaluation_metrics.csv",
        "- inference_entrypoint: HYPIR/test.py",
        "- command: $CommandText"
    ) | Set-Content -LiteralPath $MetadataPath -Encoding utf8
    Write-Host "Completed coeff_t=$CoeffT in $TotalSeconds seconds"
}
