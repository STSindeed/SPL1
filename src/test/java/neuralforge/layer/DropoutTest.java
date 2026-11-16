package neuralforge.layer;

import neuralforge.activation.Linear;
import neuralforge.activation.ReLU;
import neuralforge.activation.Sigmoid;
import neuralforge.activation.Softmax;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class DropoutTest{

    public static void main(String[] args){
        testRateValidation();
        testInferenceIsUntouched();
        testMaskValues();
        testDropFraction();
        testExpectedValueIsKept();
        testSameSeedSameMask();
        testNewMaskEveryForward();
        testZeroRateInTrainingMode();
        testBackwardUsesMask();
        testGradientWithSameMask();
        testNetworkEnableDropout();
        testToString();
        finish();
    }

    private static Matrix randomWeights(int rows, int cols, long seed){
        return Matrix.randomUniform(rows, cols, -1, 1, new Random(seed));
    }

    private static DenseLayer linearLayer(int in, int out, long seed){
        return new DenseLayer(randomWeights(in, out, seed), Matrix.zeros(1, out), new Linear());
    }

    private static double sumOf(Matrix m){
        double sum = 0.0;
        for(int i = 0; i < m.rows(); i++){
            for(int j = 0; j < m.cols(); j++){
                sum += m.get(i, j);
            }
        }
        return sum;
    }

    private static void testRateValidation(){
        DenseLayer layer = linearLayer(3, 3, 1);

        expectThrows("negative rate", () -> layer.setDropout(-0.1, new Random(1)));
        expectThrows("rate 1 would drop everything", () -> layer.setDropout(1.0, new Random(1)));
        expectThrows("rate above 1", () -> layer.setDropout(1.5, new Random(1)));
        expectThrows("rate is NaN", () -> layer.setDropout(Double.NaN, new Random(1)));
        expectThrows("positive rate without a generator", () -> layer.setDropout(0.3, null));

        layer.setDropout(0.0, null);
        check("rate 0 accepted", layer.getDropoutRate() == 0.0);

        layer.setDropout(0.3, new Random(1));
        check("rate stored", layer.getDropoutRate() == 0.3);
    }

    private static void testInferenceIsUntouched(){
        Matrix x = randomWeights(6, 3, 2);
        DenseLayer plain = linearLayer(3, 4, 3);
        DenseLayer withDropout = linearLayer(3, 4, 3);
        withDropout.setDropout(0.5, new Random(4));

        check("training is off by default", !withDropout.isTraining());
        check("same output when training is off", withDropout.forward(x).approxEquals(plain.forward(x), 0.0));
        check("no mask when training is off", withDropout.getLastMask() == null);
    }

    private static void testMaskValues(){
        DenseLayer layer = linearLayer(3, 8, 5);
        layer.setDropout(0.25, new Random(6));
        layer.setTraining(true);
        layer.forward(randomWeights(10, 3, 7));

        Matrix mask = layer.getLastMask();
        check("mask exists in training mode", mask != null);
        check("mask has the shape of the output", mask.rows() == 10 && mask.cols() == 8);

        double kept = 1.0 / 0.75;
        boolean valuesOk = true;
        for(int i = 0; i < mask.rows(); i++){
            for(int j = 0; j < mask.cols(); j++){
                double v = mask.get(i, j);
                if(v != 0.0 && !near(v, kept, 1e-12)){
                    valuesOk = false;
                }
            }
        }
        check("mask only contains 0 and 1/keep", valuesOk);
    }

    private static void testDropFraction(){
        DenseLayer layer = linearLayer(40, 100, 8);
        layer.setDropout(0.3, new Random(9));
        layer.setTraining(true);
        layer.forward(Matrix.filled(50, 40, 1.0));

        Matrix mask = layer.getLastMask();
        int dropped = 0;
        for(int i = 0; i < mask.rows(); i++){
            for(int j = 0; j < mask.cols(); j++){
                if(mask.get(i, j) == 0.0){
                    dropped++;
                }
            }
        }
        double fraction = (double) dropped / (mask.rows() * mask.cols());
        check("about 30% dropped, got " + fraction, near(fraction, 0.3, 0.03));
    }

    private static void testExpectedValueIsKept(){

        DenseLayer layer = new DenseLayer(Matrix.filled(10, 50, 0.1), Matrix.zeros(1, 50), new Linear());
        layer.setDropout(0.5, new Random(10));
        layer.setTraining(true);
        Matrix out = layer.forward(Matrix.filled(200, 10, 1.0));

        double mean = sumOf(out) / (out.rows() * out.cols());
        check("mean output stays near 1, got " + mean, near(mean, 1.0, 0.05));
    }

    private static void testSameSeedSameMask(){
        Matrix x = randomWeights(8, 3, 11);

        DenseLayer a = linearLayer(3, 6, 12);
        a.setDropout(0.4, new Random(99));
        a.setTraining(true);
        a.forward(x);

        DenseLayer b = linearLayer(3, 6, 12);
        b.setDropout(0.4, new Random(99));
        b.setTraining(true);
        b.forward(x);

        check("same seed gives the same mask", a.getLastMask().approxEquals(b.getLastMask(), 0.0));

        DenseLayer c = linearLayer(3, 6, 12);
        c.setDropout(0.4, new Random(100));
        c.setTraining(true);
        c.forward(x);
        check("another seed gives another mask", !a.getLastMask().approxEquals(c.getLastMask(), 0.0));
    }

    private static void testNewMaskEveryForward(){
        DenseLayer layer = linearLayer(3, 20, 13);
        layer.setDropout(0.5, new Random(14));
        layer.setTraining(true);
        Matrix x = randomWeights(10, 3, 15);

        layer.forward(x);
        Matrix first = layer.getLastMask();
        layer.forward(x);
        Matrix second = layer.getLastMask();
        check("a new mask for every forward pass", !first.approxEquals(second, 0.0));
    }

    private static void testZeroRateInTrainingMode(){
        Matrix x = randomWeights(6, 3, 16);
        DenseLayer plain = linearLayer(3, 4, 17);
        DenseLayer layer = linearLayer(3, 4, 17);
        layer.setDropout(0.0, null);
        layer.setTraining(true);

        check("rate 0 changes nothing", layer.forward(x).approxEquals(plain.forward(x), 0.0));
        check("rate 0 makes no mask", layer.getLastMask() == null);
    }

    private static void testBackwardUsesMask(){
        Matrix x = randomWeights(5, 3, 18);
        DenseLayer layer = linearLayer(3, 4, 19);
        layer.setDropout(0.4, new Random(20));
        layer.setTraining(true);
        layer.forward(x);

        Matrix mask = layer.getLastMask();
        Matrix ones = Matrix.filled(5, 4, 1.0);
        Matrix gradInput = layer.backward(ones);

        Matrix expectedGradZ = ones.hadamard(mask);
        check("weight gradient", layer.getGradWeights().approxEquals(x.transpose().multiply(expectedGradZ), 1e-12));
        check("bias gradient", layer.getGradBiases().approxEquals(expectedGradZ.columnSums(), 1e-12));
        check("input gradient", gradInput.approxEquals(expectedGradZ.multiply(layer.getWeights().transpose()), 1e-12));
    }

    private static void testGradientWithSameMask(){
        Matrix x = randomWeights(6, 3, 21);
        Matrix weights = randomWeights(3, 5, 22);
        Matrix biases = randomWeights(1, 5, 23);
        Matrix c = randomWeights(6, 5, 24);

        DenseLayer layer = new DenseLayer(weights.copy(), biases.copy(), new Sigmoid());
        layer.setDropout(0.4, new Random(77));
        layer.setTraining(true);
        layer.forward(x);
        layer.backward(c);
        Matrix analytical = layer.getGradWeights();

        double h = 1e-6;
        double worst = 0.0;
        for(int i = 0; i < weights.rows(); i++){
            for(int j = 0; j < weights.cols(); j++){
                double original = weights.get(i, j);

                weights.set(i, j, original + h);
                double plus = lossWithSeed(weights, biases, x, c, 77);
                weights.set(i, j, original - h);
                double minus = lossWithSeed(weights, biases, x, c, 77);
                weights.set(i, j, original);

                double numerical = (plus - minus) / (2 * h);
                worst = Math.max(worst, Math.abs(numerical - analytical.get(i, j)));
            }
        }
        check("dropout gradient matches the numerical one, worst difference " + worst, worst < 1e-7);
    }

    private static double lossWithSeed(Matrix weights, Matrix biases, Matrix x, Matrix c, long seed){
        DenseLayer layer = new DenseLayer(weights.copy(), biases.copy(), new Sigmoid());
        layer.setDropout(0.4, new Random(seed));
        layer.setTraining(true);
        return sumOf(layer.forward(x).hadamard(c));
    }

    private static void testNetworkEnableDropout(){
        Random rng = new Random(25);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(4, 6, new ReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(6, 5, new ReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(5, 3, new Softmax(), new XavierInitializer(), rng));

        net.enableDropout(0.3, new Random(26));
        check("first hidden layer has dropout", net.getLayers().get(0).getDropoutRate() == 0.3);
        check("second hidden layer has dropout", net.getLayers().get(1).getDropoutRate() == 0.3);
        check("output layer never has dropout", net.getLayers().get(2).getDropoutRate() == 0.0);

        Matrix x = randomWeights(7, 4, 27);

        net.setTraining(true);
        check("layers are in training mode", net.getLayers().get(0).isTraining() && net.getLayers().get(2).isTraining());
        Matrix first = net.forward(x);
        Matrix second = net.forward(x);
        check("training mode gives different outputs", !first.approxEquals(second, 0.0));

        net.setTraining(false);
        check("layers left training mode", !net.getLayers().get(0).isTraining());
        check("prediction mode is repeatable", net.forward(x).approxEquals(net.forward(x), 0.0));

        NeuralNetwork single = new NeuralNetwork();
        single.addLayer(new DenseLayer(4, 3, new Softmax(), new XavierInitializer(), rng));
        expectThrows("single layer network", () -> single.enableDropout(0.3, new Random(1)));

        net.enableDropout(0.0, null);
        check("dropout switched off", net.getLayers().get(0).getDropoutRate() == 0.0);
    }

    private static void testToString(){
        DenseLayer layer = new DenseLayer(4, 6, new ReLU(), new HeInitializer(), new Random(28));
        check("toString without dropout", layer.toString().equals("Dense(4 -> 6, relu)"));

        layer.setDropout(0.3, new Random(29));
        check("toString with dropout, got " + layer, layer.toString().equals("Dense(4 -> 6, relu) dropout 0.30"));
    }
}
