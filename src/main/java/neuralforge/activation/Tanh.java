package neuralforge.activation;

import neuralforge.math.Matrix;

public class Tanh implements Activation{

    @Override
    public Matrix forward(Matrix z){
        return z.apply(Math::tanh);
    }

    @Override
    public Matrix backward(Matrix z, Matrix output, Matrix gradOutput){
        Matrix derivative = output.apply(a -> 1.0 - a * a);
        return gradOutput.hadamard(derivative);
    }

    @Override
    public String name(){
        return "tanh";
    }
}
