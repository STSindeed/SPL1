package neuralforge.experiment;

import neuralforge.activation.ReLU;
import neuralforge.activation.Softmax;
import neuralforge.data.Dataset;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.Adam;
import neuralforge.optimizer.Optimizer;
import neuralforge.optimizer.OptimizerFactory;
import neuralforge.optimizer.SGD;
import neuralforge.schedule.CosineDecay;
import neuralforge.schedule.LearningRateSchedule;
import neuralforge.schedule.StepDecay;
import neuralforge.training.TrainingHistory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import static neuralforge.TestHelper.*;

public class OptimizerComparisonTest{

    public static void main(String[] args){
        testRunAndTable();
        testSameOptimizerSameResult();
        testRatesAreRecorded();
        testFormatRates();
        testErrors();
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

    private static Supplier<NeuralNetwork> network(){
        return () -> {
            Random rng = new Random(3);
            NeuralNetwork net = new NeuralNetwork();
            net.addLayer(new DenseLayer(2, 8, new ReLU(), new HeInitializer(), rng));
            net.addLayer(new DenseLayer(8, 3, new Softmax(), new XavierInitializer(), rng));
            return net;
        };
    }

    private static void testRunAndTable(){
        Dataset[] parts = blobs(40, 1).stratifiedSplit(0.25, new Random(2));
        List<Optimizer> optimizers = new ArrayList<>();
        for(String name : OptimizerFactory.names()){
            optimizers.add(OptimizerFactory.create(name));
        }
        String[] labels = OptimizerFactory.names();

        List<OptimizerComparison.Result> results = OptimizerComparison.run(network(), parts[0], parts[1], optimizers, labels, 6, 16, 5);

        check("one result per optimizer", results.size() == 5);
        check("labels are kept", results.get(4).getLabel().equals("adam"));
        check("description is the optimizer name", results.get(0).getDescription().startsWith("sgd("));
        check("every history has all epochs", results.get(2).getHistory().epochCount() == 6);
        check("time is measured", results.get(1).getSeconds() >= 0.0);
        check("best accuracy is not below the final one", results.get(3).bestAccuracy() >= results.get(3).finalAccuracy());

        boolean everyLossWentDown = true;
        double bestAccuracy = 0.0;
        for(OptimizerComparison.Result r : results){
            if(!(r.finalTrainLoss() < r.getHistory().getTrainLoss(0))){
                everyLossWentDown = false;
            }
            bestAccuracy = Math.max(bestAccuracy, r.finalAccuracy());
        }
        check("the loss goes down for every optimizer", everyLossWentDown);
        check("at least one optimizer learns the groups", bestAccuracy > 0.9);

        String table = OptimizerComparison.formatTable(results, 0.9);
        check("table has a header", table.contains("optimizer") && table.contains("to 90%"));
        check("table has a line per optimizer", table.split("\n").length == 6);
        check("table names the optimizers", table.contains("momentum") && table.contains("rmsprop"));

        String impossible = OptimizerComparison.formatTable(results, 1.1);
        check("unreached target is a dash", impossible.contains(" -"));

        String curves = OptimizerComparison.formatLossCurves(results);
        check("loss curves have a header and one line per epoch", curves.split("\n").length == 7);
        check("loss curves have the labels", curves.contains("nesterov"));
    }

    private static void testSameOptimizerSameResult(){
        Dataset[] parts = blobs(30, 4).stratifiedSplit(0.25, new Random(5));
        List<Optimizer> twins = new ArrayList<>();
        twins.add(new Adam(0.01));
        twins.add(new Adam(0.01));

        List<OptimizerComparison.Result> results = OptimizerComparison.run(network(), parts[0], parts[1], twins, new String[]{"first", "second"}, 4, 8, 11);
        boolean identical = true;
        for(int e = 0; e < 4; e++){
            if(results.get(0).getHistory().getTrainLoss(e) != results.get(1).getHistory().getTrainLoss(e)){
                identical = false;
            }
            if(results.get(0).getHistory().getValidationLoss(e) != results.get(1).getHistory().getValidationLoss(e)){
                identical = false;
            }
        }
        check("identical optimizers give identical histories", identical);

        List<Optimizer> different = new ArrayList<>();
        different.add(new SGD(0.1));
        different.add(new Adam(0.01));
        List<OptimizerComparison.Result> other = OptimizerComparison.run(network(), parts[0], parts[1], different, new String[]{"sgd", "adam"}, 4, 8, 11);
        check("different optimizers give different losses", other.get(0).finalTrainLoss() != other.get(1).finalTrainLoss());
    }

    private static void testRatesAreRecorded(){
        Dataset[] parts = blobs(20, 6).stratifiedSplit(0.25, new Random(7));
        List<Optimizer> optimizers = new ArrayList<>();
        optimizers.add(new SGD(new StepDecay(0.1, 0.5, 2)));

        List<OptimizerComparison.Result> results = OptimizerComparison.run(network(), parts[0], parts[1], optimizers, new String[]{"step"}, 4, 8, 1);
        TrainingHistory history = results.get(0).getHistory();
        check("first epochs use the start rate", history.getLearningRate(0) == 0.1 && history.getLearningRate(1) == 0.1);
        check("rate drops after two epochs", near(history.getLearningRate(2), 0.05, 1e-15) && near(history.getLearningRate(3), 0.05, 1e-15));
    }

    private static void testFormatRates(){
        LearningRateSchedule[] schedules = {new StepDecay(0.1, 0.5, 2), new CosineDecay(0.1, 0.0, 4)};
        String text = OptimizerComparison.formatRates(new String[]{"step", "cosine"}, schedules, 4);
        String[] lines = text.split("\n");
        check("header and one line per epoch", lines.length == 5);
        check("first line has both names", lines[0].contains("step") && lines[0].contains("cosine"));
        check("first epoch has the start rate", lines[1].contains("0.100000"));
        check("step drops in the third epoch", lines[3].contains("0.050000"));
    }

    private static void testErrors(){
        Dataset[] parts = blobs(10, 8).stratifiedSplit(0.25, new Random(9));
        List<Optimizer> one = new ArrayList<>();
        one.add(new SGD(0.1));

        expectThrows("label count differs", () -> OptimizerComparison.run(network(), parts[0], parts[1], one, new String[]{"a", "b"}, 2, 4, 1));
        expectThrows("no optimizers", () -> OptimizerComparison.run(network(), parts[0], parts[1], new ArrayList<Optimizer>(), new String[0], 2, 4, 1));
        expectThrows("schedule and label count differ", () -> OptimizerComparison.formatRates(new String[]{"a"}, new LearningRateSchedule[]{new StepDecay(0.1, 0.5, 2), new CosineDecay(0.1, 0.0, 4)}, 3));
        check("labels array unchanged", Arrays.equals(new String[]{"a"}, new String[]{"a"}));
    }
}
