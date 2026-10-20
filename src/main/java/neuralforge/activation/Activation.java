package neuralforge.activation;

import neuralforge.math.Matrix;

public interface Activation{
    Matrix forward(Matrix z);
    String name();
}
