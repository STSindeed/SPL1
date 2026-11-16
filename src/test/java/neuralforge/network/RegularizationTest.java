package neuralforge.network;

import neuralforge.activation.Linear;
import neuralforge.activation.Sigmoid;
import neuralforge.activation.Softmax;
import neuralforge.activation.Tanh;
import neuralforge.gradcheck.GradientChecker;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.CrossEntropy;
import neuralforge.loss.Loss;
import neuralforge.loss.MeanSquaredError;
import neuralforge.math.Matrix;
import neuralforge.optimizer.SGD;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class RegularizationTest{

    public static void main(String[] args){
        testMatrixHelpers();
        testL2Validation();
        testPenaltyValue();
        testPenaltyIgnoresBiases();
        testGradientGetsWeightDecay();
        testReportedLossIsDataLoss();
        testGradientCheckWithL2();
        testL2ShrinksWeights();
        testZeroL2ChangesNothing();
        testSnapshotAndRestore();
        testSnapshotIsIndependent();
        testSnapshotChecks();
        testSummary();
        testFitLeavesPredictionMode();
        finish();
    }

    private static NeuralNetwork smallNet(long seed){
        Random rng = new Random(seed);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(4, 5, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(5, 3, new Softmax(), new XavierInitializer(), rng));
        return net;
    }

    private static Matrix inputs(long seed){
        return Matrix.randomUniform(8, 4, -1, 1, new Random(seed));
    }

    private static Matrix targets(){
        return new Matrix(new double[][]{
            {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}
        });
    }

    private static double weightSquares(NeuralNetwork net){
        double sum = 0.0;
        for(DenseLayer layer : net.getLayers()){
            sum += layer.weightSquaredSum();
        }
        return sum;
    }

    private static void testMatrixHelpers(){
        Matrix m = new Matrix(new double[][]{{1, 2}, {3, -4}});
        check("sum of squares", near(m.sumOfSquares(), 30.0, 1e-12));

        Matrix target = Matrix.zeros(2, 2);
        Matrix sameObject = target;
        target.copyFrom(m);
        check("copyFrom copies the numbers", target.approxEquals(m, 0.0));
        check("copyFrom keeps the same object", sameObject == target);

        target.set(0, 0, 100);
        check("source is not linked to the copy", m.get(0, 0) == 1.0);

        expectThrows("copyFrom with another shape", () -> Matrix.zeros(2, 3).copyFrom(m));
    }

    private static void testL2Validation(){
        NeuralNetwork net = smallNet(1);
        check("L2 is off by default", net.getL2() == 0.0);

        expectThrows("negative L2", () -> net.setL2(-0.1));
        expectThrows("NaN L2", () -> net.setL2(Double.NaN));
        expectThrows("infinite L2", () -> net.setL2(Double.POSITIVE_INFINITY));

        net.setL2(0.25);
        check("L2 stored", net.getL2() == 0.25);
    }

    private static void testPenaltyValue(){
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(new Matrix(new double[][]{{1, 2}, {3, 4}}), new Matrix(new double[][]{{5, 6}}), new Linear()));
        net.addLayer(new DenseLayer(new Matrix(new double[][]{{1}, {1}}), new Matrix(new double[][]{{7}}), new Linear()));

        check("no penalty while L2 is off", net.regularizationPenalty() == 0.0);

        net.setL2(0.1);
        check("penalty is 0.5 * 0.1 * 32", near(net.regularizationPenalty(), 1.6, 1e-12));
    }

    private static void testPenaltyIgnoresBiases(){
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(new Matrix(new double[][]{{1}}), new Matrix(new double[][]{{1000}}), new Linear()));
        net.setL2(2.0);
        check("only weights count", near(net.regularizationPenalty(), 1.0, 1e-12));
    }

    private static void testGradientGetsWeightDecay(){
        Matrix x = inputs(2);
        double lambda = 0.3;

        NeuralNetwork plain = smallNet(3);
        NeuralNetwork decayed = smallNet(3);
        decayed.setL2(lambda);

        plain.computeGradients(x, targets(), new CrossEntropy());
        decayed.computeGradients(x, targets(), new CrossEntropy());

        for(int k = 0; k < 2; k++){
            DenseLayer a = plain.getLayers().get(k);
            DenseLayer b = decayed.getLayers().get(k);

            Matrix expected = a.getGradWeights().add(a.getWeights().scale(lambda));
            check("layer " + k + " weight gradient has lambda * W added", b.getGradWeights().approxEquals(expected, 1e-12));
            check("layer " + k + " bias gradient is unchanged", b.getGradBiases().approxEquals(a.getGradBiases(), 1e-12));
        }
    }

    private static void testReportedLossIsDataLoss(){
        Matrix x = inputs(4);
        NeuralNetwork net = smallNet(5);
        net.setL2(0.5);
        Loss loss = new CrossEntropy();

        double returned = net.computeGradients(x, targets(), loss);
        double dataLoss = net.lossOf(x, targets(), loss);
        check("returned value is the data loss", near(returned, dataLoss, 1e-12));
        check("total loss is data loss + penalty", near(net.totalLossOf(x, targets(), loss), dataLoss + net.regularizationPenalty(), 1e-12));
        check("the penalty is not zero here", net.regularizationPenalty() > 0.1);
    }

    private static void testGradientCheckWithL2(){
        double[] strengths = {0.0, 0.01, 0.5, 3.0};
        for(double lambda : strengths){
            NeuralNetwork net = smallNet(6);
            net.setL2(lambda);
            double error = GradientChecker.check(net, new CrossEntropy(), inputs(7), targets(), false);
            check("cross entropy with L2 " + lambda + ", error " + error, error < 1e-6);
        }

        Random rng = new Random(8);
        NeuralNetwork mse = new NeuralNetwork();
        mse.addLayer(new DenseLayer(3, 4, new Tanh(), new XavierInitializer(), rng));
        mse.addLayer(new DenseLayer(4, 2, new Sigmoid(), new XavierInitializer(), rng));
        mse.setL2(0.7);
        Matrix x = Matrix.randomUniform(5, 3, -1, 1, rng);
        Matrix y = Matrix.randomUniform(5, 2, 0, 1, rng);
        double error = GradientChecker.check(mse, new MeanSquaredError(), x, y, false);
        check("mse with L2, error " + error, error < 1e-6);
    }

    private static void testL2ShrinksWeights(){
        Matrix x = Matrix.randomUniform(60, 4, -1, 1, new Random(9));
        Matrix y = Matrix.zeros(60, 3);
        for(int i = 0; i < 60; i++){
            y.set(i, i % 3, 1.0);
        }

        NeuralNetwork plain = smallNet(10);
        NeuralNetwork decayed = smallNet(10);
        decayed.setL2(0.5);

        plain.fit(x, y, new CrossEntropy(), new SGD(0.1), 150, 16, new Random(11), 0);
        decayed.fit(x, y, new CrossEntropy(), new SGD(0.1), 150, 16, new Random(11), 0);

        double plainSize = weightSquares(plain);
        double decayedSize = weightSquares(decayed);
        check("L2 gives smaller weights, " + decayedSize + " vs " + plainSize, decayedSize < plainSize);
    }

    private static void testZeroL2ChangesNothing(){
        Matrix x = inputs(12);
        NeuralNetwork a = smallNet(13);
        NeuralNetwork b = smallNet(13);
        b.setL2(0.0);

        a.fit(x, targets(), new CrossEntropy(), new SGD(0.1), 20, 4, new Random(14), 0);
        b.fit(x, targets(), new CrossEntropy(), new SGD(0.1), 20, 4, new Random(14), 0);

        boolean same = true;
        for(int k = 0; k < 2; k++){
            same &= a.getLayers().get(k).getWeights().approxEquals(b.getLayers().get(k).getWeights(), 0.0);
            same &= a.getLayers().get(k).getBiases().approxEquals(b.getLayers().get(k).getBiases(), 0.0);
        }
        check("L2 = 0 gives identical training", same);
    }

    private static void testSnapshotAndRestore(){
        Matrix x = inputs(15);
        NeuralNetwork net = smallNet(16);
        Matrix before = net.forward(x);
        Matrix[] snapshot = net.snapshotParameters();
        check("two matrices per layer", snapshot.length == 4);

        net.fit(x, targets(), new CrossEntropy(), new SGD(0.5), 30, 4, new Random(17), 0);
        check("training changed the weights", !net.forward(x).approxEquals(before, 1e-9));

        net.restoreParameters(snapshot);
        check("restore brings the old predictions back", net.forward(x).approxEquals(before, 0.0));
        check("restore copies weights", net.getLayers().get(0).getWeights().approxEquals(snapshot[0], 0.0));
        check("restore copies biases", net.getLayers().get(1).getBiases().approxEquals(snapshot[3], 0.0));
    }

    private static void testSnapshotIsIndependent(){
        NeuralNetwork net = smallNet(18);
        Matrix[] snapshot = net.snapshotParameters();
        double saved = snapshot[0].get(0, 0);

        net.getLayers().get(0).getWeights().set(0, 0, saved + 5.0);
        check("snapshot did not change", snapshot[0].get(0, 0) == saved);

        Matrix weightsObject = net.getLayers().get(0).getWeights();
        net.restoreParameters(snapshot);
        check("restore keeps the matrix object", net.getLayers().get(0).getWeights() == weightsObject);
        check("restored value", weightsObject.get(0, 0) == saved);
    }

    private static void testSnapshotChecks(){
        NeuralNetwork net = smallNet(19);

        expectThrows("null snapshot", () -> net.restoreParameters(null));
        expectThrows("snapshot of another size", () -> net.restoreParameters(new Matrix[]{Matrix.zeros(1, 1)}));

        Random rng = new Random(20);
        NeuralNetwork other = new NeuralNetwork();
        other.addLayer(new DenseLayer(4, 6, new Tanh(), new XavierInitializer(), rng));
        other.addLayer(new DenseLayer(6, 3, new Softmax(), new XavierInitializer(), rng));
        expectThrows("snapshot with other shapes", () -> net.restoreParameters(other.snapshotParameters()));
    }

    private static void testSummary(){
        NeuralNetwork net = smallNet(21);
        String plain = net.summary();
        check("no L2 line without L2", !plain.contains("L2"));
        check("no dropout text without dropout", !plain.contains("dropout"));

        net.setL2(0.01);
        net.enableDropout(0.3, new Random(22));
        String full = net.summary();
        check("summary shows L2, got " + full, full.contains("L2 strength: 0.01"));
        check("summary shows dropout of the hidden layer", full.contains("dropout 0.30"));
    }

    private static void testFitLeavesPredictionMode(){
        NeuralNetwork net = smallNet(23);
        net.enableDropout(0.3, new Random(24));
        Matrix x = inputs(25);

        net.fit(x, targets(), new CrossEntropy(), new SGD(0.1), 5, 4, new Random(26), 0);

        check("layers left training mode", !net.getLayers().get(0).isTraining());
        check("predictions are repeatable after fit", net.forward(x).approxEquals(net.forward(x), 0.0));
    }
}
