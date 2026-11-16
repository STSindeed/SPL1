package neuralforge.experiment;

import neuralforge.activation.ReLU;
import neuralforge.activation.Softmax;
import neuralforge.data.Dataset;
import neuralforge.experiment.RegularizationComparison.Result;
import neuralforge.experiment.RegularizationComparison.Setup;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.Optimizer;
import neuralforge.optimizer.SGD;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import static neuralforge.TestHelper.*;

public class RegularizationComparisonTest{

    public static void main(String[] args){
        testSetupValidation();
        testSetupDescribe();
        testRunErrors();
        testResults();
        testSameSeedSameResults();
        testFormatting();
        finish();
    }

    private static Dataset blobs(int perClass, long seed){
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

    private static Supplier<NeuralNetwork> network(){
        return () -> {
            Random rng = new Random(5);
            NeuralNetwork net = new NeuralNetwork();
            net.addLayer(new DenseLayer(2, 24, new ReLU(), new HeInitializer(), rng));
            net.addLayer(new DenseLayer(24, 3, new Softmax(), new XavierInitializer(), rng));
            return net;
        };
    }

    private static Supplier<Optimizer> optimizer(){
        return () -> new SGD(0.1);
    }

    private static List<Setup> setups(){
        List<Setup> list = new ArrayList<>();
        list.add(new Setup("none", 0.0, 0.0, 0));
        list.add(new Setup("l2", 0.01, 0.0, 0));
        list.add(new Setup("dropout", 0.0, 0.3, 0));
        list.add(new Setup("early", 0.0, 0.0, 4));
        return list;
    }

    private static List<Result> runAll(){
        return RegularizationComparison.run(network(), optimizer(), blobs(12, 1), blobs(40, 2), blobs(40, 3), setups(), 200, 8, 42);
    }

    private static void testSetupValidation(){
        expectThrows("empty label", () -> new Setup("", 0.0, 0.0, 0));
        expectThrows("null label", () -> new Setup(null, 0.0, 0.0, 0));
        expectThrows("negative L2", () -> new Setup("x", -0.1, 0.0, 0));
        expectThrows("NaN L2", () -> new Setup("x", Double.NaN, 0.0, 0));
        expectThrows("dropout 1", () -> new Setup("x", 0.0, 1.0, 0));
        expectThrows("negative dropout", () -> new Setup("x", 0.0, -0.2, 0));
        expectThrows("negative patience", () -> new Setup("x", 0.0, 0.0, -1));

        Setup setup = new Setup("all", 0.01, 0.3, 5);
        check("label", setup.getLabel().equals("all"));
        check("l2", setup.getL2() == 0.01);
        check("dropout", setup.getDropout() == 0.3);
        check("patience", setup.getPatience() == 5);
    }

    private static void testSetupDescribe(){
        check("plain setup", new Setup("a", 0.0, 0.0, 0).describe().equals("no l2, no dropout, no early stopping"));
        check("full setup", new Setup("b", 0.01, 0.3, 5).describe().equals("l2=0.01, dropout=0.3, patience=5"));
    }

    private static void testRunErrors(){
        Dataset train = blobs(12, 1);
        Dataset validation = blobs(20, 2);
        List<Setup> none = new ArrayList<>();

        expectThrows("no setups", () -> RegularizationComparison.run(network(), optimizer(), train, validation, validation, none, 5, 8, 1));
        expectThrows("zero epochs", () -> RegularizationComparison.run(network(), optimizer(), train, validation, validation, setups(), 0, 8, 1));
        expectThrows("zero batch size", () -> RegularizationComparison.run(network(), optimizer(), train, validation, validation, setups(), 5, 0, 1));
    }

    private static void testResults(){
        List<Result> results = runAll();
        check("one result per setup", results.size() == 4);

        String[] labels = {"none", "l2", "dropout", "early"};
        for(int i = 0; i < results.size(); i++){
            Result r = results.get(i);
            check("label " + labels[i], r.getSetup().getLabel().equals(labels[i]));
            check("accuracies are fractions", r.trainAccuracy() >= 0 && r.trainAccuracy() <= 1 && r.testAccuracy() >= 0 && r.testAccuracy() <= 1
                && r.validationAccuracy() >= 0 && r.validationAccuracy() <= 1);
            check("gap is train minus test", near(r.gap(), r.trainAccuracy() - r.testAccuracy(), 1e-12));
            check("max epochs is stored", r.maxEpochs() == 200);
            check("test loss is positive", r.testLoss() > 0);
        }

        Result plain = results.get(0);
        Result early = results.get(3);
        check("plain run uses all epochs", plain.epochsRun() == 200 && plain.getEarlyStopping() == null);
        check("early stopping run stops sooner, ran " + early.epochsRun(), early.epochsRun() < 200);
        check("early stopping object is kept", early.getEarlyStopping() != null && early.getEarlyStopping().hasStopped());
        check("history matches epochs run", early.getHistory().epochCount() == early.epochsRun());
    }

    private static void testSameSeedSameResults(){
        List<Result> a = runAll();
        List<Result> b = runAll();
        boolean same = a.size() == b.size();
        for(int i = 0; i < a.size() && same; i++){
            same = a.get(i).testAccuracy() == b.get(i).testAccuracy()
                && a.get(i).testLoss() == b.get(i).testLoss()
                && a.get(i).epochsRun() == b.get(i).epochsRun();
        }
        check("same seed gives the same results", same);
    }

    private static void testFormatting(){
        List<Result> results = runAll();

        String table = RegularizationComparison.formatTable(results);
        check("table has a header", table.contains("setup") && table.contains("test acc") && table.contains("gap"));
        for(Result r : results){
            check("table has " + r.getSetup().getLabel(), table.contains(r.getSetup().getLabel()));
        }
        check("table shows epochs run out of the limit", table.contains("/200"));
        check("one header line plus one line per setup", table.split("\n").length == 5);

        String setups = RegularizationComparison.formatSetups(results);
        check("setups mention early stopping", setups.contains("stopped after epoch"));
        check("setups mention the settings", setups.contains("dropout=0.3") && setups.contains("l2=0.01"));

        String curves = RegularizationComparison.formatValidationLoss(results, 10);
        String[] lines = curves.split("\n");
        check("curves have a header with all labels", lines[0].contains("none") && lines[0].contains("early"));
        check("first epoch is always shown", lines[1].startsWith("1 "));
        check("a stopped run shows dashes", curves.contains("-"));
        expectThrows("step 0", () -> RegularizationComparison.formatValidationLoss(results, 0));
    }
}
