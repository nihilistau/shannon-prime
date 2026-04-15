@echo off
REM Local CUDA build helper for Windows. Sources MSVC env then compiles test_cuda.
REM The Makefile target `make test-cuda` calls nvcc directly and expects the caller's
REM shell to already have the MSVC environment (Developer Command Prompt, sourced
REM vcvars64.bat, etc.). This batch file is a convenience for running the test
REM from a stock bash shell.

setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

cd /d %~dp0\..

if not exist build mkdir build

nvcc -O2 -arch=sm_75 ^
     -o build\test_cuda.exe ^
     tests\test_cuda.c ^
     backends\cuda\shannon_prime_cuda.cu ^
     core\shannon_prime.c ^
     -lcudart

if errorlevel 1 (
    echo nvcc build FAILED
    exit /b 1
)

echo ---- RUNNING test_cuda ----
build\test_cuda.exe
exit /b %errorlevel%
