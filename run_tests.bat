@echo off
call "E:\DevelopmentTools\VisualStudio2026\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d "E:\DevelopmentTools\OmniMatchDeeplearningTool\build\qt6-debug"
ctest --output-on-failure 2>&1
