@echo off

set SRC_DIR=src\main\java
set TEST_DIR=src\test\java
set OUT_DIR=out
set MAIN_CLASS=neuralforge.Main

if not defined JAVA_OPTS set JAVA_OPTS=-Xmx2g

set TEST_CLASSES=neuralforge.math.MatrixTest neuralforge.activation.ActivationTest neuralforge.init.InitializerTest neuralforge.layer.DenseLayerTest neuralforge.loss.LossTest neuralforge.optimizer.OptimizerTest neuralforge.gradcheck.GradientCheckTest neuralforge.network.NeuralNetworkTest neuralforge.data.CsvReaderTest neuralforge.data.LabelEncoderTest neuralforge.data.DatasetTest neuralforge.data.DatasetLoaderTest neuralforge.data.IdxReaderTest neuralforge.preprocess.ScalerTest neuralforge.metrics.ConfusionMatrixTest neuralforge.metrics.MetricsTest neuralforge.training.TrainerTest neuralforge.io.ModelIOTest neuralforge.gui.PixelCanvasTest neuralforge.gui.DigitClassifierTest neuralforge.gui.DigitPanelTest neuralforge.schedule.ScheduleTest neuralforge.optimizer.OptimizerUpdateRulesTest neuralforge.optimizer.OptimizerConvergenceTest neuralforge.optimizer.OptimizerFactoryTest neuralforge.experiment.OptimizerComparisonTest

if "%1"=="clean" goto clean
if "%1"=="compile" goto compile_only
if "%1"=="test" goto test
if "%1"=="run" goto run
if "%1"=="" goto run
echo Unknown option: %1
echo Usage: build.bat [compile ^| test ^| run [arguments] ^| clean]
exit /b 1

:compile_only
call :compile
goto :eof

:run
call :compile
if errorlevel 1 exit /b 1
java %JAVA_OPTS% -cp %OUT_DIR%\main %MAIN_CLASS% %2 %3 %4 %5 %6 %7 %8 %9
goto :eof

:test
call :compile
if errorlevel 1 exit /b 1
if not exist %OUT_DIR%\test mkdir %OUT_DIR%\test
dir /s /b %TEST_DIR%\*.java > %OUT_DIR%\test_sources.txt
javac -d %OUT_DIR%\test -cp %OUT_DIR%\main @%OUT_DIR%\test_sources.txt
if errorlevel 1 exit /b 1
set FAILED=0
for %%t in (%TEST_CLASSES%) do (
    echo Running %%t
    java %JAVA_OPTS% -cp %OUT_DIR%\main;%OUT_DIR%\test %%t
    if errorlevel 1 (
        echo FAILED: %%t
        set /a FAILED+=1
    )
)
if %FAILED% GTR 0 (
    echo %FAILED% test class^(es^) failed
    exit /b 1
)
echo All test classes passed
goto :eof

:clean
if exist %OUT_DIR% rmdir /s /q %OUT_DIR%
echo Removed %OUT_DIR%
goto :eof

:compile
echo Compiling...
if not exist %OUT_DIR%\main mkdir %OUT_DIR%\main
dir /s /b %SRC_DIR%\*.java > %OUT_DIR%\sources.txt
javac -d %OUT_DIR%\main @%OUT_DIR%\sources.txt
exit /b %errorlevel%
