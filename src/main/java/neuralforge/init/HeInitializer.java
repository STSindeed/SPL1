package neuralforge.init;

import neuralforge.math.Matrix;
import java.util.Random;

public class HeInitializer implements Initializer{

    @Override
    public Matrix createWeights(int fanIn, int fanOut, Random rng){
        double std = Math.sqrt(2.0 / fanIn);
        return Matrix.randomGaussian(fanIn, fanOut, std, rng);
    }

    @Override
    public String name(){
        return "he";
    }
}
