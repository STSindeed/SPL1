package neuralforge.activation;

import neuralforge.math.Matrix;

public class Linear implements Activation{

    @Override
    public Matrix forward(Matrix z){
        return z.copy();
    }

    @Override
    public String name(){
        return "linear";
    }
}
