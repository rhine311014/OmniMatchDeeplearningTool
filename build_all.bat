@echo off
echo ============================================
echo   OmniMatch Build Script - Debug + Release
echo ============================================

call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo [ERROR] vcvarsall.bat failed
    exit /b 1
)

cd /d "E:\DevelopmentTools\OmniMatchDeeplearningTool"

echo.
echo [1/4] Configuring Debug...
cmake --preset qt6-debug 2>&1
if errorlevel 1 (
    echo [ERROR] Debug configure failed
    exit /b 1
)
echo [1/4] Debug configured OK

echo.
echo [2/4] Building Debug...
cmake --build --preset qt6-debug 2>&1
if errorlevel 1 (
    echo [WARNING] Debug build had errors
) else (
    echo [2/4] Debug build OK
)

echo.
echo [3/4] Configuring Release...
cmake --preset qt6-release 2>&1
if errorlevel 1 (
    echo [ERROR] Release configure failed
    exit /b 1
)
echo [3/4] Release configured OK

echo.
echo [4/4] Building Release...
cmake --build --preset qt6-release 2>&1
if errorlevel 1 (
    echo [WARNING] Release build had errors
) else (
    echo [4/4] Release build OK
)

echo.
echo ============================================
echo   Build Complete
echo ============================================
echo Debug output:   build\qt6-debug\
echo Release output: build\qt6-release\
