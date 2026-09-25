$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

cmake -S cpp -B cpp\build -A x64
cmake --build cpp\build --config Release --parallel
New-Item -ItemType Directory -Force native\bin | Out-Null
Copy-Item cpp\build\Release\wtivo_fast_weld.dll native\bin\wtivo_fast_weld.dll -Force
Write-Host "Native library installed at $PSScriptRoot\native\bin\wtivo_fast_weld.dll"
