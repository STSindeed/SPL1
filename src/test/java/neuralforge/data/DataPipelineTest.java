package neuralforge.data;

import neuralforge.TestData;
import neuralforge.activation.ReLU;
import neuralforge.activation.Softmax;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.CrossEntropy;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.OptimizerFactory;
import neuralforge.training.Trainer;
import java.util.Locale;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class DataPipelineTest{

    public static void main(String[] args){
        testBasicPipeline();
        testTargetSelection();
        testReproducible();
        testFitOnTrainingRowsOnly();
        testMissingLabels();
        testRegressionIsRejected();
        testNumericLabels();
        testFileProblems();
        testBadNumberMessage();
        testSemicolonFile();
        testClassWarnings();
        testNewDataGoesThroughTheSamePreprocessor();
        testNetworkLearnsFromPreparedData();
        finish();
    }

    private static String mainCsv(){
        Random rng = new Random(5);
        String[] shades = {"light", "Dark", "MID"};
        StringBuilder sb = new StringBuilder("id,a,b,shade,label\n");
        for(int i = 0; i < 60; i++){
            double a = rng.nextGaussian();
            double b = rng.nextGaussian();
            double score = a + 0.5 * b;
            String label = score > 0.5 ? "high" : (score > -0.5 ? "mid" : "low");
            String aText = i % 10 == 3 ? "?" : String.format(Locale.US, "%.4f", a);
            sb.append(i + 1).append(",").append(aText).append(",").append(String.format(Locale.US, "%.4f", b)).append(",")
                .append(shades[i % 3]).append(",").append(label).append("\n");
        }
        return sb.toString();
    }

    private static String path(String csv){
        return TestData.writeText(".csv", csv);
    }

    private static String messageOf(Runnable code){
        try{
            code.run();
        }catch(DataFormatException e){
            return e.getMessage();
        }catch(RuntimeException e){
            return "wrong exception type " + e.getClass().getName() + ": " + e.getMessage();
        }
        return null;
    }

    private static boolean reportContains(PreparedData data, String text){
        for(String message : data.getReport().getMessages()){
            if(message.contains(text)){
                return true;
            }
        }
        return false;
    }

    private static boolean allFinite(Matrix m){
        for(int i = 0; i < m.rows(); i++){
            for(int j = 0; j < m.cols(); j++){
                if(Double.isNaN(m.get(i, j)) || Double.isInfinite(m.get(i, j))){
                    return false;
                }
            }
        }
        return true;
    }

    private static void testBasicPipeline(){
        PreparedData data = DataPipeline.loadByName(path(mainCsv()), "label", new DataOptions());
        Dataset train = data.getTrain();
        Dataset test = data.getTest();

        check("all rows are used", train.size() + test.size() == 60);
        check("test share is about 20%, got " + test.size(), test.size() >= 10 && test.size() <= 14);
        check("id is dropped, so 2 numbers + 3 shades", train.featureCount() == 5 && test.featureCount() == 5);
        check("class names are sorted", java.util.Arrays.toString(data.getClassNames()).equals("[high, low, mid]"));
        check("task", data.getTask() == TaskType.MULTICLASS_CLASSIFICATION);
        check("target name", data.getTargetName().equals("label"));
        check("no holes or infinities in the features", allFinite(train.getFeatures()) && allFinite(test.getFeatures()));
        check("a profile for every column", data.getProfiles().length == 5);

        int[] trainCounts = train.classCounts();
        int[] testCounts = test.classCounts();
        boolean everyClassEverywhere = true;
        for(int c = 0; c < 3; c++){
            everyClassEverywhere &= trainCounts[c] > 0 && testCounts[c] > 0;
        }
        check("every class is in both parts (stratified)", everyClassEverywhere);

        check("report has the task", reportContains(data, "task: multiclass classification"));
        check("report has the split", reportContains(data, "split into"));
        check("report tells about the dropped column", reportContains(data, "column 'id' was dropped"));
        check("describe has the task and sizes", data.describe().contains("multiclass classification") && data.describe().contains("5 features"));
    }

    private static void testTargetSelection(){
        String file = path(mainCsv());
        DataOptions options = new DataOptions();
        PreparedData byName = DataPipeline.loadByName(file, "label", options);
        PreparedData last = DataPipeline.loadByIndex(file, true, -1, options);
        PreparedData byIndex = DataPipeline.loadByIndex(file, true, 4, options);
        check("last column by negative index", last.getTrain().getFeatures().approxEquals(byName.getTrain().getFeatures(), 0.0));
        check("same by positive index", byIndex.getTrain().getFeatures().approxEquals(byName.getTrain().getFeatures(), 0.0));

        PreparedData first = DataPipeline.loadByName(path("label,v\nx,1\ny,2\nx,3\ny,4\nx,5\ny,6\nx,7\ny,8\nx,9\ny,10\n"), "label", options);
        check("target in the first column", first.getTrain().featureCount() == 1 && first.getTask() == TaskType.BINARY_CLASSIFICATION);

        String unknown = messageOf(() -> DataPipeline.loadByName(file, "nope", options));
        check("unknown target lists the columns, got " + unknown, unknown != null && unknown.contains("no column named 'nope'") && unknown.contains("shade"));
        String outside = messageOf(() -> DataPipeline.loadByIndex(file, true, 9, options));
        check("target index outside, got " + outside, outside != null && outside.contains("does not exist") && outside.contains("5 columns"));
    }

    private static void testReproducible(){
        String file = path(mainCsv());
        PreparedData a = DataPipeline.loadByName(file, "label", new DataOptions().seed(1));
        PreparedData b = DataPipeline.loadByName(file, "label", new DataOptions().seed(1));
        PreparedData c = DataPipeline.loadByName(file, "label", new DataOptions().seed(2));
        check("same seed, same split", a.getTrain().getFeatures().approxEquals(b.getTrain().getFeatures(), 0.0)
            && a.getTest().getFeatures().approxEquals(b.getTest().getFeatures(), 0.0));
        check("other seed, other split", !a.getTrain().getFeatures().approxEquals(c.getTrain().getFeatures(), 0.0));
    }

    private static void testFitOnTrainingRowsOnly(){
        PreparedData data = DataPipeline.loadByName(path(mainCsv()), "label", new DataOptions());
        Matrix train = data.getTrain().getFeatures();
        Matrix test = data.getTest().getFeatures();

        double trainSum = 0.0;
        for(int i = 0; i < train.rows(); i++){
            trainSum += train.get(i, 1);
        }
        double testSum = 0.0;
        for(int i = 0; i < test.rows(); i++){
            testSum += test.get(i, 1);
        }
        check("training mean of b is 0", near(trainSum / train.rows(), 0.0, 1e-9));
        check("test mean of b is not forced to 0", Math.abs(testSum / test.rows()) > 1e-6);
    }

    private static void testMissingLabels(){
        StringBuilder sb = new StringBuilder("v,w,label\n");
        for(int i = 0; i < 30; i++){
            String label = i == 4 || i == 9 ? (i == 4 ? "" : "?") : (i % 2 == 0 ? "a" : "b");
            sb.append(i).append(",").append(i % 7).append(",").append(label).append("\n");
        }
        PreparedData data = DataPipeline.loadByName(path(sb.toString()), "label", new DataOptions());
        check("rows without label are dropped", data.getTrain().size() + data.getTest().size() == 28);
        check("with a warning that names the first line, got " + data.getReport().getMessages(), reportContains(data, "2 row(s) have no label")
            && reportContains(data, "line 6"));
    }

    private static void testRegressionIsRejected(){
        StringBuilder decimals = new StringBuilder("x,price\n");
        StringBuilder codes = new StringBuilder("x,code\n");
        for(int i = 0; i < 60; i++){
            decimals.append(i).append(",").append(String.format(Locale.US, "%.2f", 10 + i * 1.37)).append("\n");
            codes.append(i % 5).append(",").append(i % 30).append("\n");
        }

        String message = messageOf(() -> DataPipeline.loadByName(path(decimals.toString()), "price", new DataOptions()));
        check("decimals are rejected, got " + message, message != null && message.contains("regression") && message.contains("forceClassification"));

        String whole = messageOf(() -> DataPipeline.loadByName(path(codes.toString()), "code", new DataOptions()));
        check("30 distinct codes are rejected by default", whole != null && whole.contains("regression"));

        PreparedData forced = DataPipeline.loadByName(path(codes.toString()), "code", new DataOptions().forceClassification(true));
        check("forced classification works", forced.getTask() == TaskType.MULTICLASS_CLASSIFICATION && forced.getTrain().classCount() == 30);
        check("the report says it was forced", forced.getTaskReason().startsWith("forced by the user"));

        StringBuilder huge = new StringBuilder("x,price\n");
        for(int i = 0; i < 150; i++){
            huge.append(i).append(",").append(i + 0.5).append("\n");
        }
        String tooMany = messageOf(() -> DataPipeline.loadByName(path(huge.toString()), "price", new DataOptions().forceClassification(true)));
        check("too many classes even when forced, got " + tooMany, tooMany != null && tooMany.contains("150 distinct values"));
    }

    private static void testNumericLabels(){
        StringBuilder sb = new StringBuilder("v,label\n");
        for(int i = 0; i < 20; i++){
            sb.append(i).append(",").append(i % 2 == 0 ? (i % 4 == 0 ? "1" : "1.0") : "2.0").append("\n");
        }
        PreparedData data = DataPipeline.loadByName(path(sb.toString()), "label", new DataOptions());
        check("two classes, got " + java.util.Arrays.toString(data.getClassNames()), java.util.Arrays.toString(data.getClassNames()).equals("[1, 2]"));
        check("binary", data.getTask() == TaskType.BINARY_CLASSIFICATION);
    }

    private static void testFileProblems(){
        String missing = messageOf(() -> DataPipeline.loadByName("no/such/folder/data.csv", "label", new DataOptions()));
        check("file not found, got " + missing, missing != null && missing.contains("file not found") && missing.contains("data.csv"));

        check("duplicate header names", String.valueOf(messageOf(() -> DataPipeline.loadByIndex(path("a,b,a\n1,2,3\n"), true, 1, new DataOptions())))
            .contains("'a' twice"));
        check("empty header name", String.valueOf(messageOf(() -> DataPipeline.loadByIndex(path("a,,c\n1,2,3\n"), true, 2, new DataOptions())))
            .contains("column 2 has no name"));
        check("single column", String.valueOf(messageOf(() -> DataPipeline.loadByIndex(path("a\n1\n2\n"), true, 0, new DataOptions())))
            .contains("only one column"));

        String few = messageOf(() -> DataPipeline.loadByIndex(path("v,label\n1,a\n2,b\n3,a\n"), true, -1, new DataOptions()));
        check("too few rows, got " + few, few != null && few.contains("only 3 usable row(s)"));

        StringBuilder same = new StringBuilder("v,label\n");
        for(int i = 0; i < 20; i++){
            same.append(i).append(",yes\n");
        }
        String constant = messageOf(() -> DataPipeline.loadByName(path(same.toString()), "label", new DataOptions()));
        check("constant target, got " + constant, constant != null && constant.contains("same value ('yes')"));

        StringBuilder noLabels = new StringBuilder("v,label\n");
        for(int i = 0; i < 20; i++){
            noLabels.append(i).append(",\n");
        }
        String empty = messageOf(() -> DataPipeline.loadByName(path(noLabels.toString()), "label", new DataOptions()));
        check("no labels at all, got " + empty, empty != null && empty.contains("only 0 usable row(s)"));

        StringBuilder mixed = new StringBuilder("v,label\n");
        for(int i = 0; i < 30; i++){
            mixed.append(i).append(",").append(i == 11 ? "maybe" : String.valueOf(i % 3)).append("\n");
        }
        String mixedMessage = messageOf(() -> DataPipeline.loadByName(path(mixed.toString()), "label", new DataOptions()));
        check("word in a numeric target, got " + mixedMessage, mixedMessage != null && mixedMessage.contains("line 13") && mixedMessage.contains("'maybe'"));
    }

    private static String typoCsv(){
        StringBuilder sb = new StringBuilder("v,w,label\n");
        for(int i = 0; i < 40; i++){
            sb.append(i == 16 ? "12o" : String.valueOf(i)).append(",").append(i % 5).append(",").append(i % 2 == 0 ? "a" : "b").append("\n");
        }
        return sb.toString();
    }

    private static void testBadNumberMessage(){
        String file = path(typoCsv());
        String message = messageOf(() -> DataPipeline.loadByName(file, "label", new DataOptions()));
        check("message points at the cell, got " + message, message != null && message.contains("line 18, column 'v': '12o' is not a number"));

        PreparedData lenient = DataPipeline.loadByName(file, "label", new DataOptions().badNumbersAsMissing(true));
        check("with the option it loads", lenient.getTrain().size() + lenient.getTest().size() == 40);
        check("and warns", reportContains(lenient, "are not numbers"));
    }

    private static void testSemicolonFile(){
        String csv = "v;w;label\n1;2;a\n2;3;b\n3;4;a\n4;5;b\n5;6;a\n6;7;b\n7;8;a\n8;9;b\n9;1;a\n10;2;b\n";
        PreparedData data = DataPipeline.loadByName(path(csv), "label", new DataOptions());
        check("semicolons are detected", data.getTrain().featureCount() == 2 && data.getTrain().size() + data.getTest().size() == 10);
    }

    private static void testClassWarnings(){
        StringBuilder sb = new StringBuilder("v,label\n");
        for(int i = 0; i < 57; i++){
            sb.append(i).append(",common\n");
        }
        sb.append("100,rare\n101,rare\n102,single\n");
        PreparedData data = DataPipeline.loadByName(path(sb.toString()), "label", new DataOptions());
        check("dominant class warning", reportContains(data, "class 'common' makes up 95.0% of the rows"));
        check("rare class warning", reportContains(data, "class 'rare' has only 2 rows"));
        check("single row class warning", reportContains(data, "class 'single' has only one row"));
        check("the single row class is in the training part", data.getTrain().classCounts()[2] == 1 && data.getTest().classCounts()[2] == 0);
    }

    private static void testNewDataGoesThroughTheSamePreprocessor(){
        PreparedData data = DataPipeline.loadByName(path(mainCsv()), "label", new DataOptions());
        CsvTable fresh = CsvReader.read(path("shade,b,a\nDark,0.5,1.5\nneon,-0.5,?\n"), true);

        DataReport report = new DataReport();
        Matrix features = data.getPreprocessor().transform(fresh, null, report, "new rows");
        check("one row per new row, one column per feature", features.rows() == 2 && features.cols() == data.getTrain().featureCount());
        check("no holes", allFinite(features));
        check("a new shade is reported", report.warningCount() == 1 && report.getMessages().toString().contains("'neon'"));
    }

    private static void testNetworkLearnsFromPreparedData(){
        Random rng = new Random(11);
        String[] colors = {"red", "blue", "green"};
        StringBuilder sb = new StringBuilder("x,color,label\n");
        for(int i = 0; i < 240; i++){
            double x = rng.nextGaussian() * 2;
            String color = colors[rng.nextInt(3)];
            double score = x + (color.equals("red") ? 2.5 : color.equals("blue") ? -2.5 : 0.0);
            String xText = rng.nextDouble() < 0.08 ? "NA" : String.format(Locale.US, "%.3f", x);
            sb.append(xText).append(",").append(color).append(",").append(score > 0 ? "pos" : "neg").append("\n");
        }
        PreparedData data = DataPipeline.loadByName(path(sb.toString()), "label", new DataOptions().missingIndicators(true));
        Dataset train = data.getTrain();
        Dataset test = data.getTest();

        Random init = new Random(3);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(train.featureCount(), 12, new ReLU(), new HeInitializer(), init));
        net.addLayer(new DenseLayer(12, train.classCount(), new Softmax(), new XavierInitializer(), init));
        Trainer trainer = new Trainer(net, new CrossEntropy(), OptimizerFactory.create("adam"), new Random(4));
        trainer.train(train, test, 150, 16, 0);

        double accuracy = trainer.evaluate(test).getAccuracy();
        check("the network learns from the prepared data, accuracy " + accuracy, accuracy > 0.85);
    }
}
