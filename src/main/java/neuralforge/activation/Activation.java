package neuralforge.activation;

import neuralforge.math.Matrix;

public interface Activation{

    Matrix forward(Matrix z);
    Matrix backward(Matrix z, Matrix output, Matrix gradOutput);
    String name();
}
