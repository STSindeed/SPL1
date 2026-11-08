package neuralforge.training;

import neuralforge.TestData;
import neuralforge.activation.ReLU;
import neuralforge.activation.Softmax;
import neuralforge.data.Dataset;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.CrossEntropy;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.Optimizer;
import neuralforge.optimizer.SGD;
import neuralforge.schedule.StepDecay;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class TrainerTest{

    public static void main(String[] args){
        testTrainingLearns();
        testWithoutValidation();
        testBestEpoch();
        testEvaluateInChunks();
        testPredictionsMatchNetwork();
        testShapeChecks();
        testHistory();
        testBestEpochAndTargets();
        testScheduleIsFollowed();
        testStartEpochIsCalled();
        testSaveCsv();
        finish();
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

    private static NeuralNetwork network(long seed){
        Random rng = new Random(seed);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 8, new ReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(8, 3, new Softmax(), new XavierInitializer(), rng));
        return net;
    }

    private static Trainer trainer(NeuralNetwork net, long seed){
        return trainer(net, new SGD(0.1), seed);
    }

    private static Trainer trainer(NeuralNetwork net, Optimizer optimizer, long seed){
        return new Trainer(net, new CrossEntropy(), optimizer, new Random(seed));
    }

    private static void testTrainingLearns(){
        Dataset[] parts = blobs(60, 1).stratifiedSplit(0.25, new Random(2));
        NeuralNetwork net = network(3);
        Trainer trainer = trainer(net, 4);

        TrainingHistory history = trainer.train(parts[0], parts[1], 40, 16, 0);

        check("one entry per epoch", history.epochCount() == 40);
        check("training loss goes down", history.getTrainLoss(39) < history.getTrainLoss(0));
        check("training accuracy is high", history.getTrainAccuracy(39) > 0.9);
        check("validation accuracy is high", history.getValidationAccuracy(39) > 0.9);
        check("validation loss is a number", !Double.isNaN(history.getValidationLoss(39)));
        check("time is recorded", history.getSeconds(0) >= 0.0);

        Evaluation result = trainer.evaluate(parts[1]);
        check("final evaluation agrees with the history", near(result.getAccuracy(), history.getValidationAccuracy(39), 1e-12));
    }

    private static void testWithoutValidation(){
        Dataset data = blobs(20, 5);
        TrainingHistory history = trainer(network(6), 7).train(data, null, 5, 8, 0);

        check("validation values are NaN", Double.isNaN(history.getValidationAccuracy(0)) && Double.isNaN(history.getValidationLoss(0)));
        check("no best epoch without validation", history.bestValidationEpoch() == -1);
        check("training values are there", !Double.isNaN(history.getTrainLoss(0)));
    }

    private static void testBestEpoch(){
        TrainingHistory history = new TrainingHistory(5);
        history.record(1.0, 0.5, 1.0, 0.60, 0.1);
        history.record(0.8, 0.6, 0.9, 0.70, 0.1);
        history.record(0.6, 0.7, 0.8, 0.90, 0.1);
        history.record(0.5, 0.8, 0.8, 0.90, 0.1);
        history.record(0.4, 0.9, 0.9, 0.80, 0.1);
        check("best epoch", history.bestValidationEpoch() == 2);
        check("earlier epoch wins a tie", history.getValidationAccuracy(3) == history.getValidationAccuracy(2));
    }

    private static void testEvaluateInChunks(){
        Dataset data = blobs(834, 8);
        NeuralNetwork net = network(9);
        Trainer trainer = trainer(net, 10);
        trainer.train(data, null, 2, 64, 0);

        Evaluation result = trainer.evaluate(data);
        check("one prediction per sample", result.getPredictions().length == 2502);

        double accuracy = net.accuracy(data.getFeatures(), data.getTargets());
        double loss = net.lossOf(data.getFeatures(), data.getTargets(), new CrossEntropy());
        check("accuracy in chunks equals accuracy in one piece", near(result.getAccuracy(), accuracy, 1e-12));
        check("loss in chunks equals loss in one piece", near(result.getLoss(), loss, 1e-9));
    }

    private static void testPredictionsMatchNetwork(){
        Dataset data = blobs(30, 11);
        NeuralNetwork net = network(12);
        Evaluation result = trainer(net, 13).evaluate(data);

        int[] direct = net.predictClasses(data.getFeatures());
        boolean same = true;
        for(int i = 0; i < direct.length; i++){
            if(direct[i] != result.getPredictions()[i]){
                same = false;
            }
        }
        check("predictions are the network's predictions", same);
    }

    private static void testShapeChecks(){
        Dataset data = blobs(10, 14);
        Trainer trainer = trainer(network(15), 16);

        Dataset threeFeatures = new Dataset(new Matrix(6, 3), new int[]{0, 1, 2, 0, 1, 2}, new String[]{"a", "b", "c"});
        expectThrows("wrong number of features", () -> trainer.train(threeFeatures, null, 1, 4, 0));

        Dataset twoClasses = new Dataset(new Matrix(4, 2), new int[]{0, 1, 0, 1}, new String[]{"a", "b"});
        expectThrows("wrong number of classes", () -> trainer.evaluate(twoClasses));

        expectThrows("validation set does not fit", () -> trainer.train(data, threeFeatures, 1, 4, 0));
        expectThrows("zero epochs", () -> trainer.train(data, null, 0, 4, 0));
        expectThrows("zero batch size", () -> trainer.train(data, null, 1, 0, 0));

        Trainer empty = new Trainer(new NeuralNetwork(), new CrossEntropy(), new SGD(0.1), new Random(1));
        expectThrows("network without layers", () -> empty.evaluate(data));
    }

    private static void testHistory(){
        TrainingHistory history = new TrainingHistory(1);
        history.record(1.0, 0.5, Double.NaN, Double.NaN, 0.1);
        check("rate is NaN when it was not given", Double.isNaN(history.getLearningRate(0)));
        expectThrows("history is full", () -> history.record(1.0, 0.5, 1.0, 0.5, 0.1));
        expectThrows("epoch too big", () -> history.getTrainLoss(1));
        expectThrows("negative epoch", () -> history.getTrainLoss(-1));
        expectThrows("no room for epochs", () -> new TrainingHistory(0));
    }

    private static void testBestEpochAndTargets(){
        TrainingHistory history = new TrainingHistory(5);
        history.record(1.0, 0.5, 1.0, 0.60, 0.1, 0.1);
        history.record(0.8, 0.6, 0.9, 0.70, 0.1, 0.1);
        history.record(0.6, 0.7, 0.8, 0.90, 0.1, 0.05);
        history.record(0.5, 0.8, 0.8, 0.90, 0.1, 0.05);
        history.record(0.4, 0.9, 0.9, 0.80, 0.1, 0.025);

        check("learning rate is stored", history.getLearningRate(0) == 0.1 && history.getLearningRate(4) == 0.025);
        check("first epoch above 65%", history.firstEpochReaching(0.65) == 2);
        check("first epoch with exactly 90%", history.firstEpochReaching(0.90) == 3);
        check("target never reached", history.firstEpochReaching(0.95) == -1);

        TrainingHistory noValidation = new TrainingHistory(2);
        noValidation.record(1.0, 0.5, Double.NaN, Double.NaN, 0.1);
        check("no validation, no epoch", noValidation.firstEpochReaching(0.1) == -1);
    }

    private static void testScheduleIsFollowed(){
        Dataset data = blobs(20, 20);
        Optimizer optimizer = new SGD(new StepDecay(0.1, 0.5, 2));
        TrainingHistory history = trainer(network(21), optimizer, 22).train(data, null, 5, 8, 0);

        check("epochs 1 and 2 use the start rate", history.getLearningRate(0) == 0.1 && history.getLearningRate(1) == 0.1);
        check("epochs 3 and 4 use half", near(history.getLearningRate(2), 0.05, 1e-15) && near(history.getLearningRate(3), 0.05, 1e-15));
        check("epoch 5 uses a quarter", near(history.getLearningRate(4), 0.025, 1e-15));
    }

    private static void testStartEpochIsCalled(){
        final List<Integer> seen = new ArrayList<>();
        final SGD inner = new SGD(0.1);
        Optimizer recorder = new Optimizer(){
            @Override
            public void update(neuralforge.layer.DenseLayer layer){
                inner.update(layer);
            }

            @Override
            public String name(){
                return "recorder";
            }

            @Override
            public void startEpoch(int epoch){
                seen.add(epoch);
            }

            @Override
            public double learningRate(){
                return 0.1;
            }
        };

        trainer(network(23), recorder, 24).train(blobs(10, 25), null, 4, 8, 0);
        check("startEpoch called once per epoch, in order", seen.equals(java.util.Arrays.asList(0, 1, 2, 3)));

        Optimizer plain = new Optimizer(){
            @Override
            public void update(neuralforge.layer.DenseLayer layer){
                inner.update(layer);
            }

            @Override
            public String name(){
                return "plain";
            }
        };
        TrainingHistory history = trainer(network(26), plain, 27).train(blobs(10, 28), null, 2, 8, 0);
        check("unknown rate is NaN in the history", Double.isNaN(history.getLearningRate(0)));
    }

    private static void testSaveCsv(){
        Dataset data = blobs(10, 17);
        TrainingHistory history = trainer(network(18), 19).train(data, null, 3, 8, 0);

        try{
            String path = TestData.writeText(".csv", "");
            history.saveCsv(path);
            List<String> lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);

            check("header and one line per epoch", lines.size() == 4);
            check("header", lines.get(0).startsWith("epoch,learning_rate,train_loss"));
            check("first epoch", lines.get(1).startsWith("1,"));
            check("learning rate column", lines.get(1).split(",", -1)[1].equals("0.10000000"));
            check("validation cells are empty", lines.get(1).split(",", -1)[4].isEmpty() && lines.get(1).split(",", -1)[5].isEmpty());
            check("seven columns", lines.get(1).split(",", -1).length == 7);
        }catch(IOException e){
            check("could not read the csv back: " + e.getMessage(), false);
        }
    }
}
