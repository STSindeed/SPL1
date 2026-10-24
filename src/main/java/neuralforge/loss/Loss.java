package neuralforge.loss;

import neuralforge.math.Matrix;

public interface Loss{

    double compute(Matrix predicted, Matrix target);

    Matrix gradient(Matrix predicted, Matrix target);

    String name();
}
