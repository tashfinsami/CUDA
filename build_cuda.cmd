@echo off
REM Set MSVC environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Compile the CUDA file passed as first argument
nvcc -G -g "%~1" -o "%~dpn1.exe"

REM Run the program
"%~dpn1.exe"

pause
