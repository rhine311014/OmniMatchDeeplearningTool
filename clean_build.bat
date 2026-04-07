@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "E:\DevelopmentTools\OmniMatchDeeplearningTool"
if exist build\qt6-debug rd /s /q build\qt6-debug
cmake --preset qt6-debug
if errorlevel 1 exit /b 1
cmake --build --preset qt6-debug
if errorlevel 1 exit /b 1
