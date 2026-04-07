call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\DevelopmentTools\OmniMatchDeeplearningTool
echo STARTING CMAKE CONFIGURE
cmake --preset qt6-debug
echo DONE WITH EXIT CODE %ERRORLEVEL%
