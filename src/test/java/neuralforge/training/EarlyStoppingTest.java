package neuralforge.training;

import neuralforge.activation.ReLU;
import neuralforge.activation.Softmax;
import neuralforge.data.Dataset;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.CrossEntropy;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.SGD;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class EarlyStoppingTest{

    public static void main(String[] args){
        testValidation();
        testStopsAfterPatience();
        testPatienceStartsOverAfterImprovement();
        testMinDelta();
        testAccuracyMonitor();
        testNaN();
        testRestoreBest();
        testRestoreSwitchedOff();
        testNothingToRestore();
        testReset();
        testDescribe();
        testTrainerStopsEarly();
        testTrainerWithoutEarlyStopping();
        testTrainerNeedsValidation();
        testStopperCanBeReused();
        testTrainerWithDropout();
        testEvaluateIgnoresDropout();
        finish();
    }

    private static NeuralNetwork tinyNet(double weight){
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(Matrix.filled(2, 2, weight), Matrix.filled(1, 2, weight), new Softmax()));
        return net;
    }

    private static int stopEpochForLosses(EarlyStopping rule, double[] losses){
        NeuralNetwork net = tinyNet(0.5);
        for(int i = 0; i < losses.length; i++){
            if(rule.update(i + 1, losses[i], 0.0, net)){
                return i + 1;
            }
        }
        return 0;
    }

    private static void testValidation(){
        expectThrows("patience 0", () -> new EarlyStopping(0));
        expectThrows("negative patience", () -> new EarlyStopping(-3));
        expectThrows("no monitor", () -> new EarlyStopping(null, 3, 0.0, true));
        expectThrows("negative min delta", () -> new EarlyStopping(EarlyStopping.Monitor.LOSS, 3, -0.1, true));
        expectThrows("NaN min delta", () -> new EarlyStopping(EarlyStopping.Monitor.LOSS, 3, Double.NaN, true));

        EarlyStopping rule = new EarlyStopping(4);
        check("default monitors the loss", rule.getMonitor() == EarlyStopping.Monitor.LOSS);
        check("default patience", rule.getPatience() == 4);
        check("default min delta", rule.getMinDelta() == 0.0);
        check("default restores the best weights", rule.isRestoringBest());
        check("not stopped at the start", !rule.hasStopped());
    }

    private static void testStopsAfterPatience(){
        EarlyStopping rule = new EarlyStopping(2);
        double[] losses = {1.0, 0.8, 0.7, 0.75, 0.74, 0.9};

        check("stops at epoch 5", stopEpochForLosses(rule, losses) == 5);
        check("best epoch is 3", rule.getBestEpoch() == 3);
        check("best value is 0.7", rule.getBestValue() == 0.7);
        check("it knows it stopped", rule.hasStopped());
        check("stopped epoch is 5", rule.getStoppedEpoch() == 5);
    }

    private static void testPatienceStartsOverAfterImprovement(){
        EarlyStopping rule = new EarlyStopping(3);

        double[] losses = {1.0, 1.1, 1.2, 0.9, 1.0, 1.0, 1.0};
        check("stops at epoch 7", stopEpochForLosses(rule, losses) == 7);
        check("best epoch is 4", rule.getBestEpoch() == 4);

        EarlyStopping neverStops = new EarlyStopping(3);
        double[] improving = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5};
        check("keeps going while it improves", stopEpochForLosses(neverStops, improving) == 0);
        check("best is the last epoch", neverStops.getBestEpoch() == 6);
    }

    private static void testMinDelta(){
        EarlyStopping rule = new EarlyStopping(EarlyStopping.Monitor.LOSS, 2, 0.01, true);
        double[] losses = {1.0, 0.995, 0.99};
        check("tiny improvements do not count", stopEpochForLosses(rule, losses) == 3);
        check("best epoch stays at 1", rule.getBestEpoch() == 1);

        EarlyStopping noDelta = new EarlyStopping(EarlyStopping.Monitor.LOSS, 2, 0.0, true);
        check("without min delta the same numbers keep improving", stopEpochForLosses(noDelta, losses) == 0);
    }

    private static void testAccuracyMonitor(){
        EarlyStopping rule = new EarlyStopping(EarlyStopping.Monitor.ACCURACY, 2, 0.0, true);
        NeuralNetwork net = tinyNet(0.5);
        double[] accuracy = {0.5, 0.6, 0.6, 0.55};

        int stoppedAt = 0;
        for(int i = 0; i < accuracy.length && stoppedAt == 0; i++){

            if(rule.update(i + 1, 1.0 - 0.1 * i, accuracy[i], net)){
                stoppedAt = i + 1;
            }
        }
        check("stops at epoch 4", stoppedAt == 4);
        check("best epoch is 2", rule.getBestEpoch() == 2);
        check("best value is the accuracy", rule.getBestValue() == 0.6);
    }

    private static void testNaN(){
        EarlyStopping rule = new EarlyStopping(2);
        double[] losses = {1.0, Double.NaN, Double.NaN};
        check("NaN epochs run out the patience", stopEpochForLosses(rule, losses) == 3);
        check("best epoch is before the NaN", rule.getBestEpoch() == 1);

        EarlyStopping late = new EarlyStopping(5);
        NeuralNetwork net = tinyNet(0.5);
        late.update(1, Double.NaN, 0.0, net);
        late.update(2, 0.5, 0.0, net);
        check("first real number is the best", late.getBestEpoch() == 2 && late.getBestValue() == 0.5);
    }

    private static void testRestoreBest(){
        EarlyStopping rule = new EarlyStopping(2);
        NeuralNetwork net = tinyNet(1.0);

        rule.update(1, 0.5, 0.0, net);

        net.getLayers().get(0).getWeights().set(0, 0, 9.0);
        net.getLayers().get(0).getBiases().set(0, 1, 7.0);
        rule.update(2, 0.9, 0.0, net);
        net.getLayers().get(0).getWeights().set(1, 1, -3.0);
        rule.update(3, 1.2, 0.0, net);

        check("it stopped", rule.hasStopped());
        check("restore says it did something", rule.restoreBest(net));
        check("weights are back", net.getLayers().get(0).getWeights().approxEquals(Matrix.filled(2, 2, 1.0), 0.0));
        check("biases are back", net.getLayers().get(0).getBiases().approxEquals(Matrix.filled(1, 2, 1.0), 0.0));
    }

    private static void testRestoreSwitchedOff(){
        EarlyStopping rule = new EarlyStopping(EarlyStopping.Monitor.LOSS, 1, 0.0, false);
        NeuralNetwork net = tinyNet(1.0);
        rule.update(1, 0.5, 0.0, net);
        net.getLayers().get(0).getWeights().set(0, 0, 9.0);
        rule.update(2, 0.9, 0.0, net);

        check("restore does nothing when switched off", !rule.restoreBest(net));
        check("weights are left alone", net.getLayers().get(0).getWeights().get(0, 0) == 9.0);
    }

    private static void testNothingToRestore(){
        EarlyStopping rule = new EarlyStopping(3);
        check("no epoch recorded, nothing to restore", !rule.restoreBest(tinyNet(1.0)));
    }

    private static void testReset(){
        EarlyStopping rule = new EarlyStopping(1);
        NeuralNetwork net = tinyNet(1.0);
        rule.update(1, 0.5, 0.0, net);
        rule.update(2, 0.6, 0.0, net);
        check("stopped before the reset", rule.hasStopped());

        rule.reset();
        check("not stopped after reset", !rule.hasStopped());
        check("best epoch cleared", rule.getBestEpoch() == 0);
        check("stopped epoch cleared", rule.getStoppedEpoch() == 0);
        check("best value cleared", Double.isNaN(rule.getBestValue()));
        check("saved weights cleared", !rule.restoreBest(net));
    }

    private static void testDescribe(){
        String text = new EarlyStopping(5).describe();
        check("describe mentions the loss, got " + text, text.contains("validation loss"));
        check("describe mentions the patience", text.contains("patience 5"));
        check("describe mentions restoring", text.contains("restores best weights"));
        check("accuracy version", new EarlyStopping(EarlyStopping.Monitor.ACCURACY, 2, 0.0, false).describe().contains("validation accuracy"));
    }

    private static Dataset noisyBlobs(int perClass, long seed){
        Random rng = new Random(seed);
        double[][] centers = {{-1, -1}, {1, -1}, {0, 1.2}};
        Matrix x = new Matrix(perClass * 3, 2);
        int[] labels = new int[perClass * 3];
        for(int c = 0; c < 3; c++){
            for(int k = 0; k < perClass; k++){
                int row = c * perClass + k;
                x.set(row, 0, centers[c][0] + rng.nextGaussian() * 1.2);
                x.set(row, 1, centers[c][1] + rng.nextGaussian() * 1.2);
                labels[row] = c;
            }
        }
        return new Dataset(x, labels, new String[]{"a", "b", "c"});
    }

    private static NeuralNetwork bigNet(long seed){
        Random rng = new Random(seed);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 32, new ReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(32, 3, new Softmax(), new XavierInitializer(), rng));
        return net;
    }

    private static void testTrainerStopsEarly(){
        Dataset train = noisyBlobs(12, 1);
        Dataset validation = noisyBlobs(60, 2);

        NeuralNetwork net = bigNet(3);
        Trainer trainer = new Trainer(net, new CrossEntropy(), new SGD(0.1), new Random(4));
        EarlyStopping rule = new EarlyStopping(5);
        trainer.setEarlyStopping(rule);
        check("trainer returns the rule", trainer.getEarlyStopping() == rule);

        TrainingHistory history = trainer.train(train, validation, 600, 8, 0);

        check("it stopped before the epoch limit, ran " + history.epochCount(), rule.hasStopped() && history.epochCount() < 600);
        check("it ran exactly up to the stopping epoch", history.epochCount() == rule.getStoppedEpoch());
        check("it waited exactly the patience", rule.getStoppedEpoch() - rule.getBestEpoch() == 5);

        double lowest = Double.MAX_VALUE;
        for(int e = 0; e < history.epochCount(); e++){
            lowest = Math.min(lowest, history.getValidationLoss(e));
        }
        check("best epoch has the lowest validation loss", near(history.getValidationLoss(rule.getBestEpoch() - 1), lowest, 1e-12));

        double finalLoss = trainer.evaluate(validation).getLoss();
        check("network has the best weights, " + finalLoss + " vs " + lowest, near(finalLoss, lowest, 1e-9));
    }

    private static void testTrainerWithoutEarlyStopping(){
        Dataset train = noisyBlobs(12, 5);
        Dataset validation = noisyBlobs(20, 6);
        Trainer trainer = new Trainer(bigNet(7), new CrossEntropy(), new SGD(0.1), new Random(8));

        TrainingHistory history = trainer.train(train, validation, 25, 8, 0);
        check("all epochs are used", history.epochCount() == 25);
    }

    private static void testTrainerNeedsValidation(){
        Dataset train = noisyBlobs(12, 9);
        Trainer trainer = new Trainer(bigNet(10), new CrossEntropy(), new SGD(0.1), new Random(11));
        trainer.setEarlyStopping(new EarlyStopping(3));

        expectThrows("early stopping without validation data", () -> trainer.train(train, null, 10, 8, 0));

        trainer.setEarlyStopping(null);
        check("training without a rule works", trainer.train(train, null, 3, 8, 0).epochCount() == 3);
    }

    private static void testStopperCanBeReused(){
        Dataset train = noisyBlobs(12, 12);
        Dataset validation = noisyBlobs(60, 13);
        EarlyStopping rule = new EarlyStopping(4);

        Trainer first = new Trainer(bigNet(14), new CrossEntropy(), new SGD(0.1), new Random(15));
        first.setEarlyStopping(rule);
        int firstRun = first.train(train, validation, 600, 8, 0).epochCount();

        Trainer second = new Trainer(bigNet(14), new CrossEntropy(), new SGD(0.1), new Random(15));
        second.setEarlyStopping(rule);
        int secondRun = second.train(train, validation, 600, 8, 0).epochCount();

        check("same setup gives the same number of epochs, " + firstRun + " and " + secondRun, firstRun == secondRun);
    }

    private static Dataset blobs(int perClass, long seed){
        Random rng = new Random(seed);
        double[][] centers = {{-2, -2}, {2, -2}, {0, 2.5}};
        Matrix x = new Matrix(perClass * 3, 2);
        int[] labels = new int[perClass * 3];
        for(int c = 0; c < 3; c++){
            for(int k = 0; k < perClass; k++){
                int row = c * perClass + k;
                x.set(row, 0, centers[c][0] + rng.nextGaussian() * 0.7);
                x.set(row, 1, centers[c][1] + rng.nextGaussian() * 0.7);
                labels[row] = c;
            }
        }
        return new Dataset(x, labels, new String[]{"a", "b", "c"});
    }

    private static void testTrainerWithDropout(){
        Dataset[] parts = blobs(60, 16).stratifiedSplit(0.25, new Random(17));
        NeuralNetwork net = bigNet(18);
        net.enableDropout(0.3, new Random(19));
        Trainer trainer = new Trainer(net, new CrossEntropy(), new SGD(0.1), new Random(20));

        TrainingHistory history = trainer.train(parts[0], parts[1], 60, 16, 0);

        double accuracy = history.getValidationAccuracy(history.epochCount() - 1);
        check("learns with dropout, accuracy " + accuracy, accuracy > 0.9);
        check("hidden layer is not left in training mode", !net.getLayers().get(0).isTraining());

        Matrix x = parts[1].getFeatures();
        check("predictions are repeatable after training", net.forward(x).approxEquals(net.forward(x), 0.0));
    }

    private static void testEvaluateIgnoresDropout(){
        Dataset data = blobs(20, 21);
        NeuralNetwork net = bigNet(22);
        net.enableDropout(0.5, new Random(23));
        Trainer trainer = new Trainer(net, new CrossEntropy(), new SGD(0.1), new Random(24));

        net.setTraining(true);
        double first = trainer.evaluate(data).getLoss();
        net.setTraining(true);
        double second = trainer.evaluate(data).getLoss();
        check("evaluate gives the same loss every time", first == second);
        check("evaluate leaves prediction mode on", !net.getLayers().get(0).isTraining());
    }
}
