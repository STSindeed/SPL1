package neuralforge.gradcheck;

import neuralforge.activation.Activation;
import neuralforge.activation.LeakyReLU;
import neuralforge.activation.Linear;
import neuralforge.activation.Sigmoid;
import neuralforge.activation.Softmax;
import neuralforge.activation.Tanh;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.CrossEntropy;
import neuralforge.loss.MeanSquaredError;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class GradientCheckTest{

    private static final double LIMIT = 1e-6;

    public static void main(String[] args){
        testRelativeError();
        testTanhSigmoidMse();
        testLeakyReluSoftmaxCrossEntropy();
        testDeepNetwork();
        testLinearRegression();
        testDetectsWrongGradient();
        finish();
    }

    private static void testRelativeError(){
        check("equal numbers", GradientChecker.relativeError(0.5, 0.5) == 0.0);
        check("both zero", GradientChecker.relativeError(0.0, 0.0) == 0.0);

        check("ten percent apart", near(GradientChecker.relativeError(1.0, 1.1), 0.1 / 2.1, 1e-12));

        check("tiny numbers are not a false alarm", GradientChecker.relativeError(1e-12, 3e-12) < 1e-3);
    }

    private static void testTanhSigmoidMse(){
        Random rng = new Random(21);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(3, 4, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(4, 2, new Sigmoid(), new XavierInitializer(), rng));

        Matrix x = Matrix.randomUniform(5, 3, -1, 1, rng);
        Matrix y = Matrix.randomUniform(5, 2, 0, 1, rng);

        double error = GradientChecker.check(net, new MeanSquaredError(), x, y, false);
        check("tanh + sigmoid + mse, error " + error, error < LIMIT);
    }

    private static void testLeakyReluSoftmaxCrossEntropy(){
        Random rng = new Random(22);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(4, 5, new LeakyReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(5, 3, new Softmax(), new XavierInitializer(), rng));

        Matrix x = Matrix.randomUniform(6, 4, -1, 1, rng);
        Matrix y = new Matrix(new double[][]{
            {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 0, 1}, {0, 1, 0}
        });

        double error = GradientChecker.check(net, new CrossEntropy(), x, y, false);
        check("leaky relu + softmax + cross entropy, error " + error, error < LIMIT);
    }

    private static void testDeepNetwork(){
        Random rng = new Random(23);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(3, 6, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(6, 5, new Sigmoid(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(5, 4, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(4, 3, new Softmax(), new XavierInitializer(), rng));

        Matrix x = Matrix.randomUniform(4, 3, -1, 1, rng);
        Matrix y = new Matrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 0}});

        double error = GradientChecker.check(net, new CrossEntropy(), x, y, false);
        check("deep network, error " + error, error < LIMIT);
    }

    private static void testLinearRegression(){
        Random rng = new Random(24);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 5, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(5, 1, new Linear(), new XavierInitializer(), rng));

        Matrix x = Matrix.randomUniform(8, 2, -2, 2, rng);
        Matrix y = Matrix.randomUniform(8, 1, -3, 3, rng);

        double error = GradientChecker.check(net, new MeanSquaredError(), x, y, false);
        check("linear regression, error " + error, error < LIMIT);
    }

    private static void testDetectsWrongGradient(){
        Random rng = new Random(25);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(3, 4, new BrokenSigmoid(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(4, 2, new Sigmoid(), new XavierInitializer(), rng));

        Matrix x = Matrix.randomUniform(5, 3, -1, 1, rng);
        Matrix y = Matrix.randomUniform(5, 2, 0, 1, rng);

        double error = GradientChecker.check(net, new MeanSquaredError(), x, y, false);
        check("checker catches a wrong backward(), error " + error, error > 1e-3);
    }

    private static class BrokenSigmoid implements Activation{
        private final Sigmoid real = new Sigmoid();

        @Override
        public Matrix forward(Matrix z){
            return real.forward(z);
        }

        @Override
        public Matrix backward(Matrix z, Matrix output, Matrix gradOutput){
            return real.backward(z, output, gradOutput).scale(2.0);
        }

        @Override
        public String name(){
            return "broken_sigmoid";
        }
    }
}
