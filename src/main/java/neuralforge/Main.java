package neuralforge;

import neuralforge.activation.ReLU;
import neuralforge.activation.Softmax;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.CrossEntropy;
import neuralforge.loss.Loss;
import neuralforge.math.Matrix;
import java.util.Random;

public class Main{

    public static void main(String[] args){
        System.out.println("NeuralForge - forward pass demo");
        System.out.println();

        Random rng = new Random(1);

        DenseLayer hidden = new DenseLayer(4, 5, new ReLU(), new HeInitializer(), rng);
        DenseLayer output = new DenseLayer(5, 3, new Softmax(), new XavierInitializer(), rng);
        System.out.println("layer 1: " + hidden + ", parameters: " + hidden.parameterCount());
        System.out.println("layer 2: " + output + ", parameters: " + output.parameterCount());
        System.out.println();

        Matrix x = new Matrix(new double[][]{
                {0.5, -1.2, 3.0, 0.7},
                {1.5, 0.3, -0.8, 2.1}
        });

        Matrix probabilities = output.forward(hidden.forward(x));
        System.out.println("class probabilities (each row adds up to 1):");
        System.out.print(probabilities);

        Matrix target = new Matrix(new double[][]{
                {0, 0, 1},
                {1, 0, 0}
        });

        Loss loss = new CrossEntropy();
        System.out.println();
        System.out.println(loss.name() + " = " + loss.compute(probabilities, target));

        System.out.println("pure guessing with 3 classes would give " + Math.log(3));
    }
}
