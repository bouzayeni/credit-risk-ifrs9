# Dot-source before running uv in this repo:  . .\dev.ps1
# Your user environment points UV_PROJECT_ENVIRONMENT at the e-commerce project's venv;
# this overrides it for the current PowerShell session only.
$env:UV_PROJECT_ENVIRONMENT = "K:\credit-risk-data\venv"
$env:UV_CACHE_DIR = "K:\smart-ecommerce-intelligence-data\uv-cache"
$env:UV_LINK_MODE = "copy"
Write-Host "uv environment: $env:UV_PROJECT_ENVIRONMENT"
