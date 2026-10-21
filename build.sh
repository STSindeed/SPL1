#!/bin/bash
# Build script for Linux, macOS and Git Bash.
# Usage: ./build.sh [compile | test | run | clean]
# Running it with no argument does the same as "run" (compile + run the demo).

# where things live (these paths follow the src/main/java layout)
SRC_DIR=src/main/java
TEST_DIR=src/test/java
OUT_DIR=out
MAIN_CLASS=neuralforge.Main

# full names of the test classes to run, separated by spaces
# add every new test class here
TEST_CLASSES="neuralforge.math.MatrixTest neuralforge.activation.ActivationTest neuralforge.init.InitializerTest neuralforge.layer.DenseLayerTest neuralforge.loss.LossTest"

# compile all main source files into out/main
compile(){
    echo "Compiling..."
    mkdir -p $OUT_DIR/main
    # find lists every .java file under src/main/java (also in sub-folders)
    # "|| exit 1" stops the script if javac reports an error
    javac -d $OUT_DIR/main $(find $SRC_DIR -name "*.java") || exit 1
}

# compile the test files (they need the compiled main classes on the classpath)
# and then run every test class
run_tests(){
    mkdir -p $OUT_DIR/test
    javac -d $OUT_DIR/test -cp $OUT_DIR/main $(find $TEST_DIR -name "*.java") || exit 1
    for t in $TEST_CLASSES; do
        echo "Running $t"
        # classpath has both folders; ":" separates them on Linux/macOS
        java -cp $OUT_DIR/main:$OUT_DIR/test $t || exit 1
    done
}

# start the program
run_main(){
    java -cp $OUT_DIR/main $MAIN_CLASS
}

# $1 is the first argument given to the script
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
        # delete all compiled output
        rm -rf $OUT_DIR
        echo "Removed $OUT_DIR"
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: ./build.sh [compile | test | run | clean]"
        exit 1
        ;;
esac
