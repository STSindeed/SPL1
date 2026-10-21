@echo off
REM Build script for Windows (cmd).
REM Usage: build.bat [compile | test | run | clean]
REM Running it with no argument does the same as "run" (compile + run the demo).

REM where things live (these paths follow the src\main\java layout)
set SRC_DIR=src\main\java
set TEST_DIR=src\test\java
set OUT_DIR=out
set MAIN_CLASS=neuralforge.Main

REM full names of the test classes to run, separated by spaces
REM add every new test class here
set TEST_CLASSES=neuralforge.math.MatrixTest neuralforge.activation.ActivationTest neuralforge.init.InitializerTest neuralforge.layer.DenseLayerTest neuralforge.loss.LossTest

REM pick what to do from the first argument (%1)
if "%1"=="clean" goto clean
if "%1"=="compile" goto compile_only
if "%1"=="test" goto test
if "%1"=="run" goto run
if "%1"=="" goto run
echo Unknown option: %1
echo Usage: build.bat [compile ^| test ^| run ^| clean]
exit /b 1

:compile_only
call :compile
goto :eof

:run
call :compile
if errorlevel 1 exit /b 1
java -cp %OUT_DIR%\main %MAIN_CLASS%
goto :eof

:test
call :compile
if errorlevel 1 exit /b 1
if not exist %OUT_DIR%\test mkdir %OUT_DIR%\test
REM write the list of test files into a text file, javac reads it with @
dir /s /b %TEST_DIR%\*.java > %OUT_DIR%\test_sources.txt
javac -d %OUT_DIR%\test -cp %OUT_DIR%\main @%OUT_DIR%\test_sources.txt
if errorlevel 1 exit /b 1
REM run each test class, ";" separates classpath folders on Windows
for %%t in (%TEST_CLASSES%) do (
    echo Running %%t
    java -cp %OUT_DIR%\main;%OUT_DIR%\test %%t
    if errorlevel 1 exit /b 1
)
goto :eof

:clean
if exist %OUT_DIR% rmdir /s /q %OUT_DIR%
echo Removed %OUT_DIR%
goto :eof

:compile
echo Compiling...
if not exist %OUT_DIR%\main mkdir %OUT_DIR%\main
REM dir /s /b gives the full path of every .java file, one per line
dir /s /b %SRC_DIR%\*.java > %OUT_DIR%\sources.txt
javac -d %OUT_DIR%\main @%OUT_DIR%\sources.txt
exit /b %errorlevel%
