package neuralforge;

import neuralforge.activation.ReLU;
import neuralforge.activation.Sigmoid;
import neuralforge.activation.Softmax;
import neuralforge.activation.Tanh;
import neuralforge.gradcheck.GradientChecker;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.CrossEntropy;
import neuralforge.loss.MeanSquaredError;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.SGD;
import java.util.Locale;
import java.util.Random;

public class Main{

    public static void main(String[] args){
        System.out.println("NeuralForge - training demo");
        System.out.println();

        gradientCheckDemo();
        xorDemo();
        blobsDemo();
    }


    private static void gradientCheckDemo(){
        System.out.println("== 1. gradient check ==");

        Random rng = new Random(1);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(4, 5, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(5, 3, new Softmax(), new XavierInitializer(), rng));

        Matrix x = Matrix.randomUniform(6, 4, -1, 1, rng);
        Matrix y = new Matrix(new double[][]{
            {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}
        });

        double worst = GradientChecker.check(net, new CrossEntropy(), x, y, true);
        System.out.println(String.format(Locale.US, "worst relative error: %.3e", worst));
        System.out.println();
    }


    private static void xorDemo(){
        System.out.println("== 2. XOR ==");

        Matrix x = new Matrix(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Matrix y = new Matrix(new double[][]{{0}, {1}, {1}, {0}});

        Random rng = new Random(7);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 4, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(4, 1, new Sigmoid(), new XavierInitializer(), rng));
        System.out.print(net.summary());

        net.fit(x, y, new MeanSquaredError(), new SGD(0.8), 3000, 4, rng, 500);

        Matrix out = net.forward(x);
        System.out.println("input -> prediction (target)");
        for(int i = 0; i < x.rows(); i++){
            System.out.println(String.format(Locale.US, "(%.0f, %.0f) -> %.4f  (%.0f)",
                x.get(i, 0), x.get(i, 1), out.get(i, 0), y.get(i, 0)));
        }
        System.out.println();
    }

    private static void blobsDemo(){
        System.out.println("== 3. three groups of points ==");

        Random rng = new Random(8);
        int perClass = 40;

        double[][] centers = {{-2, -2}, {2, -2}, {0, 2.5}};
        Matrix x = new Matrix(perClass * 3, 2);
        Matrix y = new Matrix(perClass * 3, 3);
        for(int c = 0; c < 3; c++){
            for(int k = 0; k < perClass; k++){
                int row = c * perClass + k;
                x.set(row, 0, centers[c][0] + rng.nextGaussian() * 0.7);
                x.set(row, 1, centers[c][1] + rng.nextGaussian() * 0.7);
                y.set(row, c, 1.0);
            }
        }

        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 8, new ReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(8, 3, new Softmax(), new XavierInitializer(), rng));
        System.out.print(net.summary());

        System.out.println(String.format(Locale.US, "accuracy before training: %.1f%%", 100 * net.accuracy(x, y)));

        net.fit(x, y, new CrossEntropy(), new SGD(0.1), 100, 16, rng, 20);

        System.out.println(String.format(Locale.US, "accuracy after training:  %.1f%%", 100 * net.accuracy(x, y)));
    }
}
