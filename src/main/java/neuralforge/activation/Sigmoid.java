package neuralforge.activation;

import neuralforge.math.Matrix;

public class Sigmoid implements Activation{

    @Override
    public Matrix forward(Matrix z){
        return z.apply(Sigmoid::sigmoid);
    }

    public static double sigmoid(double x){
        if(x >= 0){
            return 1.0 / (1.0 + Math.exp(-x));
        }else{
            double e = Math.exp(x);
            return e / (1.0 + e);
        }
    }

    @Override
    public Matrix backward(Matrix z, Matrix output, Matrix gradOutput){
        Matrix derivative = output.apply(a -> a * (1.0 - a));
        return gradOutput.hadamard(derivative);
    }

    @Override
    public String name(){
        return "sigmoid";
    }
}
