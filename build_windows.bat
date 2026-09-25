@echo off
setlocal
cd /d "%~dp0"
if not defined CMAKE_GENERATOR set "CMAKE_GENERATOR=Visual Studio 17 2022"
echo [WTiVo] Configuring multithreaded native CPU weld...
cmake -S cpp -B build -G "%CMAKE_GENERATOR%" -A x64
if errorlevel 1 exit /b %errorlevel%
cmake --build build --config Release
if errorlevel 1 exit /b %errorlevel%
if not exist native mkdir native
copy /Y "build\Release\wtivo_fast_weld_mt.dll" "native\wtivo_fast_weld_mt.dll"
if errorlevel 1 exit /b %errorlevel%
echo.
echo [WTiVo] Native DLL installed in:
echo %~dp0native\wtivo_fast_weld_mt.dll
endlocal
