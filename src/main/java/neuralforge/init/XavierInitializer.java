package neuralforge.init;

import neuralforge.math.Matrix;
import java.util.Random;

public class XavierInitializer implements Initializer{

    @Override
    public Matrix createWeights(int fanIn, int fanOut, Random rng){
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        return Matrix.randomUniform(fanIn, fanOut, -limit, limit, rng);
    }

    @Override
    public String name(){
        return "xavier";
    }
}
