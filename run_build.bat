@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "E:\DevelopmentTools\OmniMatchDeeplearningTool"
echo === CMAKE CONFIGURE ===
cmake --preset qt6-debug
if errorlevel 1 (
    echo CMAKE CONFIGURE FAILED
    exit /b 1
)
echo === CMAKE BUILD ===
cmake --build --preset qt6-debug 2>&1
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
echo === BUILD SUCCESS ===
