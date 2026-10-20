package neuralforge.activation;

import neuralforge.math.Matrix;

public class Tanh implements Activation{

    @Override
    public Matrix forward(Matrix z){
        return z.apply(Math::tanh);
    }

    @Override
    public String name(){
        return "tanh";
    }
}
