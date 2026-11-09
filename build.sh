#!/bin/bash

SRC_DIR=src/main/java
TEST_DIR=src/test/java
OUT_DIR=out
MAIN_CLASS=neuralforge.Main

JAVA_OPTS=${JAVA_OPTS:--Xmx2g}

TEST_CLASSES="neuralforge.math.MatrixTest neuralforge.activation.ActivationTest neuralforge.init.InitializerTest neuralforge.layer.DenseLayerTest neuralforge.loss.LossTest neuralforge.optimizer.OptimizerTest neuralforge.gradcheck.GradientCheckTest neuralforge.network.NeuralNetworkTest neuralforge.data.CsvReaderTest neuralforge.data.LabelEncoderTest neuralforge.data.DatasetTest neuralforge.data.DatasetLoaderTest neuralforge.data.IdxReaderTest neuralforge.preprocess.ScalerTest neuralforge.metrics.ConfusionMatrixTest neuralforge.metrics.MetricsTest neuralforge.training.TrainerTest neuralforge.io.ModelIOTest neuralforge.gui.PixelCanvasTest neuralforge.gui.DigitClassifierTest neuralforge.gui.DigitPanelTest neuralforge.schedule.ScheduleTest neuralforge.optimizer.OptimizerUpdateRulesTest neuralforge.optimizer.OptimizerConvergenceTest neuralforge.optimizer.OptimizerFactoryTest neuralforge.experiment.OptimizerComparisonTest"

compile(){
    echo "Compiling..."
    mkdir -p "$OUT_DIR/main"
    javac -d "$OUT_DIR/main" $(find "$SRC_DIR" -name "*.java") || exit 1
}

run_tests(){
    mkdir -p "$OUT_DIR/test"
    javac -d "$OUT_DIR/test" -cp "$OUT_DIR/main" $(find "$TEST_DIR" -name "*.java") || exit 1

    failed=0
    for t in $TEST_CLASSES; do
        echo "Running $t"
        java $JAVA_OPTS -cp "$OUT_DIR/main:$OUT_DIR/test" $t || {
            echo "FAILED: $t"
            failed=$((failed + 1))
        }
    done

    if [ $failed -gt 0 ]; then
        echo "$failed test class(es) failed"
        exit 1
    fi
    echo "All test classes passed"
}

run_main(){
    java $JAVA_OPTS -cp "$OUT_DIR/main" $MAIN_CLASS "$@"
}

command="$1"
shift

case "$command" in
    compile)
        compile
        ;;
    test)
        compile
        run_tests
        ;;
    run|"")
        compile
        run_main "$@"
        ;;
    clean)
        rm -rf "$OUT_DIR"
        echo "Removed $OUT_DIR"
        ;;
    *)
        echo "Unknown option: $command"
        echo "Usage: ./build.sh [compile | test | run [arguments] | clean]"
        exit 1
        ;;
esac
