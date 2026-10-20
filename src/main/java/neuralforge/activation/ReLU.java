package neuralforge.activation;

import neuralforge.math.Matrix;

public class ReLU implements Activation{

    @Override
    public Matrix forward(Matrix z){
        return z.apply(x -> x > 0 ? x : 0.0);
    }

    @Override
    public String name(){
        return "relu";
    }
}
