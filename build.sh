#!/bin/bash

SRC_DIR=src/main/java
TEST_DIR=src/test/java
OUT_DIR=out
MAIN_CLASS=neuralforge.Main


TEST_CLASSES="neuralforge.math.MatrixTest neuralforge.activation.ActivationTest neuralforge.init.InitializerTest neuralforge.layer.DenseLayerTest neuralforge.loss.LossTest neuralforge.optimizer.OptimizerTest neuralforge.gradcheck.GradientCheckTest neuralforge.network.NeuralNetworkTest"

compile(){
    echo "Compiling..."
    mkdir -p $OUT_DIR/main
    javac -d $OUT_DIR/main $(find $SRC_DIR -name "*.java") || exit 1
}

run_tests(){
    mkdir -p $OUT_DIR/test
    javac -d $OUT_DIR/test -cp $OUT_DIR/main $(find $TEST_DIR -name "*.java") || exit 1
    for t in $TEST_CLASSES; do
        echo "Running $t"
        java -cp $OUT_DIR/main:$OUT_DIR/test $t || exit 1
    done
}

run_main(){
    java -cp $OUT_DIR/main $MAIN_CLASS
}

case "$1" in
    compile)
        compile
        ;;
    test)
        compile
        run_tests
        ;;
    run|"")
        compile
        run_main
        ;;
    clean)
        rm -rf $OUT_DIR
        echo "Removed $OUT_DIR"
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: ./build.sh [compile | test | run | clean]"
        exit 1
        ;;
esac
