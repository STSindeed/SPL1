package neuralforge.preprocess;

import neuralforge.math.Matrix;

public interface Scaler{

    void fit(Matrix data);

    Matrix transform(Matrix data);

    String name();

    default Matrix fitTransform(Matrix data){
        fit(data);
        return transform(data);
    }
}
