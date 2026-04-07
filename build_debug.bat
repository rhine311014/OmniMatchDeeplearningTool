@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d "E:\DevelopmentTools\OmniMatchDeeplearningTool"
cmake --preset qt6-debug
if errorlevel 1 (
    echo CMAKE CONFIGURE FAILED
    exit /b 1
)
cmake --build --preset qt6-debug 2>&1
