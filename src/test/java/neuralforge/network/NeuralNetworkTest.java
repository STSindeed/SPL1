package neuralforge.network;

import neuralforge.activation.Linear;
import neuralforge.activation.ReLU;
import neuralforge.activation.Sigmoid;
import neuralforge.activation.Softmax;
import neuralforge.activation.Tanh;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.CrossEntropy;
import neuralforge.loss.Loss;
import neuralforge.loss.MeanSquaredError;
import neuralforge.math.Matrix;
import neuralforge.optimizer.SGD;
import java.util.Arrays;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class NeuralNetworkTest{

    public static void main(String[] args){
        testBuilding();
        testForward();
        testShuffle();
        testLossGoesDown();
        testFitHistory();
        testLearnsXor();
        testLearnsClasses();
        testErrors();
        finish();
    }

    private static void testBuilding(){
        Random rng = new Random(1);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(4, 5, new ReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(5, 3, new Softmax(), new XavierInitializer(), rng));

        check("layer count", net.getLayers().size() == 2);

        check("parameter count", net.parameterCount() == 43);
        check("summary mentions total", net.summary().contains("total parameters: 43"));

        expectThrows("layer sizes do not fit", () -> net.addLayer(new DenseLayer(6, 2, new Linear(), new XavierInitializer(), rng)));
    }

    private static void testForward(){
        Random rng = new Random(2);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(3, 4, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(4, 2, new Softmax(), new XavierInitializer(), rng));

        Matrix out = net.forward(Matrix.randomUniform(7, 3, -1, 1, rng));
        check("forward shape", out.shape().equals("7x2"));
        check("forward rows are probabilities", near(out.get(0, 0) + out.get(0, 1), 1.0, 1e-9));
    }

    private static void testShuffle(){
        int[] a = new int[50];
        for(int i = 0; i < a.length; i++){
            a[i] = i;
        }
        int[] original = a.clone();
        NeuralNetwork.shuffle(a, new Random(3));

        int[] sorted = a.clone();
        Arrays.sort(sorted);
        check("shuffle keeps every element once", Arrays.equals(sorted, original));
        check("shuffle changes the order", !Arrays.equals(a, original));

        int[] b = original.clone();
        NeuralNetwork.shuffle(b, new Random(3));
        check("shuffle is reproducible", Arrays.equals(a, b));

        NeuralNetwork.shuffle(new int[0], new Random(1));
        NeuralNetwork.shuffle(new int[]{7}, new Random(1));
        check("shuffle tiny arrays", true);
    }

    private static void testLossGoesDown(){
        Random rng = new Random(4);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(3, 6, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(6, 2, new Sigmoid(), new XavierInitializer(), rng));

        Matrix x = Matrix.randomUniform(10, 3, -1, 1, rng);
        Matrix y = Matrix.randomUniform(10, 2, 0, 1, rng);
        Loss loss = new MeanSquaredError();
        SGD sgd = new SGD(0.5);

        double before = net.lossOf(x, y, loss);
        for(int i = 0; i < 200; i++){
            net.trainBatch(x, y, loss, sgd);
        }
        double after = net.lossOf(x, y, loss);
        check("loss decreases (" + before + " -> " + after + ")", after < before);

        NeuralNetwork other = new NeuralNetwork();
        other.addLayer(new DenseLayer(3, 2, new Linear(), new XavierInitializer(), new Random(9)));
        double expected = other.lossOf(x, y, loss);
        check("trainBatch returns the loss before the step", near(other.trainBatch(x, y, loss, sgd), expected, 1e-12));
    }

    private static void testFitHistory(){
        Random rng = new Random(5);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 4, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(4, 1, new Linear(), new XavierInitializer(), rng));

        Matrix x = Matrix.randomUniform(10, 2, -1, 1, rng);
        Matrix y = Matrix.randomUniform(10, 1, -1, 1, rng);

        double[] history = net.fit(x, y, new MeanSquaredError(), new SGD(0.1), 30, 3, rng, 0);
        check("history has one entry per epoch", history.length == 30);
        check("history values are finite", !Double.isNaN(history[29]) && !Double.isInfinite(history[29]));
        check("loss lower at the end", history[29] < history[0]);
    }

    private static void testLearnsXor(){
        Matrix x = new Matrix(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Matrix y = new Matrix(new double[][]{{0}, {1}, {1}, {0}});

        Random rng = new Random(7);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 4, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(4, 1, new Sigmoid(), new XavierInitializer(), rng));

        net.fit(x, y, new MeanSquaredError(), new SGD(0.8), 3000, 4, rng, 0);

        Matrix out = net.forward(x);
        check("xor (0,0) -> 0", out.get(0, 0) < 0.2);
        check("xor (0,1) -> 1", out.get(1, 0) > 0.8);
        check("xor (1,0) -> 1", out.get(2, 0) > 0.8);
        check("xor (1,1) -> 0", out.get(3, 0) < 0.2);
    }

    private static void testLearnsClasses(){
        Random rng = new Random(8);
        int perClass = 30;
        double[][] centers = {{-2, -2}, {2, -2}, {0, 2.5}};
        Matrix x = new Matrix(perClass * 3, 2);
        Matrix y = new Matrix(perClass * 3, 3);
        for(int c = 0; c < 3; c++){
            for(int k = 0; k < perClass; k++){
                int row = c * perClass + k;
                x.set(row, 0, centers[c][0] + rng.nextGaussian() * 0.6);
                x.set(row, 1, centers[c][1] + rng.nextGaussian() * 0.6);
                y.set(row, c, 1.0);
            }
        }

        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 8, new ReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(8, 3, new Softmax(), new XavierInitializer(), rng));

        double before = net.accuracy(x, y);
        net.fit(x, y, new CrossEntropy(), new SGD(0.1), 100, 16, rng, 0);
        double after = net.accuracy(x, y);

        check("accuracy is a fraction", after >= 0.0 && after <= 1.0);
        check("accuracy after training is high (" + after + ")", after > 0.95);
        check("accuracy improved (" + before + " -> " + after + ")", after > before);
        check("predictClasses length", net.predictClasses(x).length == perClass * 3);
    }

    private static void testErrors(){
        NeuralNetwork empty = new NeuralNetwork();
        expectThrows("forward without layers", () -> empty.forward(Matrix.zeros(1, 2)));

        Random rng = new Random(1);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 2, new Linear(), new XavierInitializer(), rng));
        Matrix x = Matrix.zeros(4, 2);

        expectThrows("fit sample count mismatch", () -> net.fit(x, Matrix.zeros(3, 2), new MeanSquaredError(), new SGD(0.1), 1, 2, rng, 0));
        expectThrows("fit zero epochs", () -> net.fit(x, x, new MeanSquaredError(), new SGD(0.1), 0, 2, rng, 0));
        expectThrows("fit zero batch size", () -> net.fit(x, x, new MeanSquaredError(), new SGD(0.1), 1, 0, rng, 0));
    }
}
