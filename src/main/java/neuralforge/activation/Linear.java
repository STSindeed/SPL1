package neuralforge.activation;

import neuralforge.math.Matrix;

public class Linear implements Activation{

    @Override
    public Matrix forward(Matrix z){
        return z.copy();
    }

    @Override
    public Matrix backward(Matrix z, Matrix output, Matrix gradOutput){
        return gradOutput.copy();
    }

    @Override
    public String name(){
        return "linear";
    }
}
