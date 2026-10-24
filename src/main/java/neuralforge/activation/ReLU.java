package neuralforge.activation;

import neuralforge.math.Matrix;

public class ReLU implements Activation{

    @Override
    public Matrix forward(Matrix z){
        return z.apply(x -> x > 0 ? x : 0.0);
    }

    @Override
    public Matrix backward(Matrix z, Matrix output, Matrix gradOutput){
        Matrix derivative = z.apply(x -> x > 0 ? 1.0 : 0.0);
        return gradOutput.hadamard(derivative);
    }

    @Override
    public String name(){
        return "relu";
    }
}
