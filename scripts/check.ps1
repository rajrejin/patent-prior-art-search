# Quality gate (PowerShell equivalent of `make check`): lint, types, tests with the 80 percent
# coverage gate. Stops at the first failing step and returns its exit code.
$ErrorActionPreference = "Continue"
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$steps = @(
    @("-m", "ruff", "check", "."),
    @("-m", "ruff", "format", "--check", "."),
    @("-m", "mypy"),
    @("-m", "pytest", "-rs", "--cov=src/patentrag", "--cov=api", "--cov-fail-under=80")
)
foreach ($s in $steps) {
    & $py @s
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $($s -join ' ')"; exit $LASTEXITCODE }
}
Write-Host "quality gate passed"
