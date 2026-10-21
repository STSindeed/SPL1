package neuralforge.init;

import neuralforge.math.Matrix;
import java.util.Random;

public interface Initializer{
    Matrix createWeights(int fanIn, int fanOut, Random rng);
    String name();
}
