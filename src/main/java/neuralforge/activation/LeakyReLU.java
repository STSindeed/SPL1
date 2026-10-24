package neuralforge.activation;

import neuralforge.math.Matrix;

public class LeakyReLU implements Activation{

    private final double slope;

    public LeakyReLU(){
        this(0.01);
    }

    public LeakyReLU(double slope){
        this.slope = slope;
    }

    @Override
    public Matrix forward(Matrix z){
        double s = this.slope;
        return z.apply(x -> x > 0 ? x : s * x);
    }

    @Override
    public Matrix backward(Matrix z, Matrix output, Matrix gradOutput){
        double s = this.slope;
        Matrix derivative = z.apply(x -> x > 0 ? 1.0 : s);
        return gradOutput.hadamard(derivative);
    }

    @Override
    public String name(){
        return "leaky_relu";
    }
}
