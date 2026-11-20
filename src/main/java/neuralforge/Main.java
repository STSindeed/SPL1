package neuralforge;

import neuralforge.activation.ReLU;
import neuralforge.activation.Sigmoid;
import neuralforge.activation.Softmax;
import neuralforge.activation.Tanh;
import neuralforge.cli.Cli;
import neuralforge.cli.Suggestions;
import neuralforge.data.ColumnProfile;
import neuralforge.data.DataFormatException;
import neuralforge.data.DataOptions;
import neuralforge.data.DataPipeline;
import neuralforge.data.Dataset;
import neuralforge.data.DatasetLoader;
import neuralforge.data.PreparedData;
import neuralforge.experiment.OptimizerComparison;
import neuralforge.experiment.RegularizationComparison;
import neuralforge.gradcheck.GradientChecker;
import neuralforge.gui.DigitApp;
import neuralforge.init.HeInitializer;
import neuralforge.init.XavierInitializer;
import neuralforge.io.ModelIO;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.CrossEntropy;
import neuralforge.loss.MeanSquaredError;
import neuralforge.math.Matrix;
import neuralforge.metrics.ConfusionMatrix;
import neuralforge.metrics.Metrics;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.Optimizer;
import neuralforge.optimizer.OptimizerFactory;
import neuralforge.optimizer.SGD;
import neuralforge.preprocess.StandardScaler;
import neuralforge.schedule.LearningRateSchedule;
import neuralforge.schedule.ScheduleFactory;
import neuralforge.training.EarlyStopping;
import neuralforge.training.Evaluation;
import neuralforge.training.Trainer;
import neuralforge.training.TrainingHistory;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.function.Supplier;

public class Main{

    private static final String USAGE = "usage: Main [all | check | xor | blobs | iris | mnist [trainLimit [epochs [optimizer [schedule [l2 [dropout [patience]]]]]]]\n"
        + "                  | optimizers [iris | mnist [trainLimit [epochs]]] | decay [trainLimit [epochs [optimizer [startRate]]]]\n"
        + "                  | regularize [trainLimit [epochs [l2 [dropout [patience]]]]]\n"
        + "                  | data [csvFile [targetColumn [epochs]]]\n"
        + "                  | eval [modelFile] | gui [modelFile]]\n"
        + "       Main train <config> | predict <model> <csv> | evaluate <model> <csv> | inspect <model> | template | help [command]\n"
        + "optimizers: sgd, momentum, nesterov, rmsprop, adam\n"
        + "schedules: constant, step, exp, inverse, cosine";

    private static final String MODEL_PATH = "models/mnist.nfm";

    private static final String MNIST_FOLDER = "data/mnist/";

    public static void main(String[] args){
        if(args.length > 0 && Cli.isCommand(args[0])){
            int code = Cli.run(args, System.out, System.err);
            if(code != Cli.OK){
                System.exit(code);
            }
            return;
        }

        String mode = args.length > 0 ? args[0].toLowerCase() : "all";
        boolean all = mode.equals("all");

        System.out.println("NeuralForge - training demo");
        System.out.println();

        boolean known = all;
        if(all || mode.equals("check")){
            gradientCheckDemo();
            known = true;
        }
        if(all || mode.equals("xor")){
            xorDemo();
            known = true;
        }
        if(all || mode.equals("blobs")){
            blobsDemo();
            known = true;
        }
        if(all || mode.equals("iris")){
            irisDemo();
            known = true;
        }
        if(mode.equals("mnist")){
            mnistDemo(args);
            known = true;
        }
        if(mode.equals("optimizers")){
            optimizersDemo(args);
            known = true;
        }
        if(mode.equals("decay")){
            decayDemo(args);
            known = true;
        }
        if(mode.equals("regularize")){
            regularizeDemo(args);
            known = true;
        }
        if(mode.equals("data")){
            dataDemo(args);
            known = true;
        }
        if(mode.equals("eval")){
            evalDemo(args);
            known = true;
        }
        if(mode.equals("gui")){
            guiDemo(args);
            known = true;
        }

        if(!known){
            List<String> names = new ArrayList<>(Arrays.asList("all", "check", "xor", "blobs", "iris", "mnist", "optimizers", "decay", "regularize", "data", "eval", "gui"));
            names.addAll(Cli.commandNames());
            System.out.println("unknown demo or command '" + mode + "'" + Suggestions.hint(mode, names));
            System.out.println(USAGE);
        }
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

        net.setL2(0.1);
        double worstWithL2 = GradientChecker.check(net, new CrossEntropy(), x, y, false);
        System.out.println(String.format(Locale.US, "worst relative error with L2 0.1: %.3e", worstWithL2));
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
        System.out.println();
    }

    private static void irisDemo(){
        System.out.println("== 4. iris (csv pipeline) ==");

        String path = "data/iris.csv";
        if(!new File(path).exists()){
            System.out.println("file " + path + " not found, run the program from the project folder");
            System.out.println();
            return;
        }

        Random rng = new Random(42);
        Dataset[] parts = prepareIris(rng);
        Dataset train = parts[0];
        Dataset test = parts[1];

        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(train.featureCount(), 8, new ReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(8, train.classCount(), new Softmax(), new XavierInitializer(), rng));
        System.out.print(net.summary());

        Trainer trainer = new Trainer(net, new CrossEntropy(), new SGD(0.1), rng);
        TrainingHistory history = trainer.train(train, test, 100, 16, 20);
        history.saveCsv("results/iris_history.csv");

        Evaluation result = trainer.evaluate(test);
        ConfusionMatrix matrix = ConfusionMatrix.of(test.getLabels(), result.getPredictions(), test.classCount());
        System.out.print(matrix.format(test.getClassNames()));
        System.out.print(matrix.report(test.getClassNames()));
        System.out.println();
    }

    private static Dataset[] prepareIris(Random rng){
        Dataset all = DatasetLoader.fromCsv("data/iris.csv", true, -1);
        System.out.print(all.describe());

        Dataset[] parts = all.stratifiedSplit(0.2, rng);

        StandardScaler scaler = new StandardScaler();
        Dataset train = parts[0].withFeatures(scaler.fitTransform(parts[0].getFeatures()));
        Dataset test = parts[1].withFeatures(scaler.transform(parts[1].getFeatures()));
        System.out.println("train: " + train.size() + " samples, test: " + test.size() + " samples");
        return new Dataset[]{train, test};
    }

    private static void mnistDemo(String[] args){
        System.out.println("== 5. mnist ==");

        int trainLimit = 0;
        int epochs = 5;
        double l2 = 0.0;
        double dropout = 0.0;
        int patience = 0;
        try{
            if(args.length > 1){
                trainLimit = Integer.parseInt(args[1]);
            }
            if(args.length > 2){
                epochs = Integer.parseInt(args[2]);
            }
            if(args.length > 5){
                l2 = Double.parseDouble(args[5]);
            }
            if(args.length > 6){
                dropout = Double.parseDouble(args[6]);
            }
            if(args.length > 7){
                patience = Integer.parseInt(args[7]);
            }
        }catch(NumberFormatException e){
            System.out.println(USAGE);
            return;
        }
        if(l2 < 0.0 || !(dropout >= 0.0 && dropout < 1.0) || patience < 0){
            System.out.println("l2 must be >= 0, dropout must be in [0, 1) and patience must be >= 0");
            System.out.println(USAGE);
            return;
        }

        String optimizerName = args.length > 3 ? args[3] : "sgd";
        String scheduleName = args.length > 4 ? args[4] : "constant";
        Optimizer optimizer;
        try{
            double rate = OptimizerFactory.defaultRate(optimizerName);
            LearningRateSchedule schedule = ScheduleFactory.create(scheduleName, rate, epochs);
            optimizer = OptimizerFactory.create(optimizerName, schedule);
        }catch(IllegalArgumentException e){
            System.out.println(e.getMessage());
            System.out.println(USAGE);
            return;
        }

        String[] files = findMnistFiles();
        if(files == null){
            return;
        }

        long loadStart = System.nanoTime();
        Dataset train = DatasetLoader.fromIdx(files[0], files[1], trainLimit, 10);
        Dataset test = DatasetLoader.fromIdx(files[2], files[3], 0, 10);
        double loadSeconds = (System.nanoTime() - loadStart) / 1e9;
        System.out.println(String.format(Locale.US, "loaded %d training and %d test images in %.1fs", train.size(), test.size(), loadSeconds));

        Random rng = new Random(2026);

        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(train.featureCount(), 128, new ReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(128, train.classCount(), new Softmax(), new XavierInitializer(), rng));
        net.setL2(l2);
        if(dropout > 0.0){
            net.enableDropout(dropout, new Random(2027));
        }
        System.out.print(net.summary());

        System.out.println("optimizer: " + optimizer.name());
        Trainer trainer = new Trainer(net, new CrossEntropy(), optimizer, rng);

        Dataset fitSet = train;
        Dataset watchSet = test;
        if(patience > 0){

            Dataset[] split = train.stratifiedSplit(0.1, rng);
            fitSet = split[0];
            watchSet = split[1];
            trainer.setEarlyStopping(new EarlyStopping(patience));
            System.out.println("early stopping with patience " + patience + ", watching " + watchSet.size() + " held-back training images");
        }
        TrainingHistory history = trainer.train(fitSet, watchSet, epochs, 64, 1);
        history.saveCsv("results/mnist_history.csv");

        ModelIO.save(net, MODEL_PATH);
        System.out.println("model saved to " + MODEL_PATH);

        int best = history.bestValidationEpoch();
        String watchedName = patience > 0 ? "validation" : "test";
        System.out.println(String.format(Locale.US, "best %s accuracy %.2f%% at epoch %d", watchedName, 100 * history.getValidationAccuracy(best), best + 1));

        Evaluation result = trainer.evaluate(test);
        System.out.println(String.format(Locale.US, "final test accuracy %.2f%%", 100 * result.getAccuracy()));
        ConfusionMatrix matrix = ConfusionMatrix.of(test.getLabels(), result.getPredictions(), test.classCount());
        System.out.print(matrix.format(test.getClassNames()));
        System.out.print(matrix.report(test.getClassNames()));
        System.out.println();
    }

    private static void optimizersDemo(String[] args){
        System.out.println("== 6. optimizers ==");

        String data = args.length > 1 ? args[1].toLowerCase() : "mnist";
        int trainLimit = 10000;
        int epochs = data.equals("iris") ? 60 : 5;
        try{
            if(args.length > 2){
                trainLimit = Integer.parseInt(args[2]);
            }
            if(args.length > 3){
                epochs = Integer.parseInt(args[3]);
            }
        }catch(NumberFormatException e){
            System.out.println(USAGE);
            return;
        }

        List<Optimizer> optimizers = new ArrayList<>();
        for(String name : OptimizerFactory.names()){
            optimizers.add(OptimizerFactory.create(name));
        }
        String[] labels = OptimizerFactory.names();

        if(data.equals("iris")){
            if(!new File("data/iris.csv").exists()){
                System.out.println("file data/iris.csv not found, run the program from the project folder");
                return;
            }
            Dataset[] parts = prepareIris(new Random(42));

            Supplier<NeuralNetwork> iris = () -> {
                Random rng = new Random(7);
                NeuralNetwork net = new NeuralNetwork();
                net.addLayer(new DenseLayer(4, 8, new ReLU(), new HeInitializer(), rng));
                net.addLayer(new DenseLayer(8, 3, new Softmax(), new XavierInitializer(), rng));
                return net;
            };
            compare(iris, parts[0], parts[1], optimizers, labels, epochs, 16, 0.9);
            return;
        }

        String[] files = findMnistFiles();
        if(files == null){
            return;
        }
        Dataset train = DatasetLoader.fromIdx(files[0], files[1], trainLimit, 10);
        Dataset test = DatasetLoader.fromIdx(files[2], files[3], 0, 10);
        System.out.println(train.size() + " training images, " + test.size() + " test images, " + epochs + " epochs");

        compare(mnistNetwork(), train, test, optimizers, labels, epochs, 64, 0.92);
    }

    private static void decayDemo(String[] args){
        System.out.println("== 7. learning rate decay ==");

        int trainLimit = 10000;
        int epochs = 10;

        String optimizerName = args.length > 3 ? args[3] : "adam";
        double rate;
        try{
            if(args.length > 1){
                trainLimit = Integer.parseInt(args[1]);
            }
            if(args.length > 2){
                epochs = Integer.parseInt(args[2]);
            }
            rate = args.length > 4 ? Double.parseDouble(args[4]) : 0.01;
        }catch(IllegalArgumentException e){
            System.out.println(e.getMessage());
            System.out.println(USAGE);
            return;
        }

        String[] labels = ScheduleFactory.names();
        LearningRateSchedule[] schedules = new LearningRateSchedule[labels.length];
        List<Optimizer> optimizers = new ArrayList<>();
        for(int i = 0; i < labels.length; i++){
            schedules[i] = ScheduleFactory.create(labels[i], rate, epochs);
            optimizers.add(OptimizerFactory.create(optimizerName, schedules[i]));
        }

        System.out.println(optimizerName + ", learning rate in every epoch, starting at " + rate);
        System.out.print(OptimizerComparison.formatRates(labels, schedules, epochs));
        System.out.println();

        String[] files = findMnistFiles();
        if(files == null){
            return;
        }
        Dataset train = DatasetLoader.fromIdx(files[0], files[1], trainLimit, 10);
        Dataset test = DatasetLoader.fromIdx(files[2], files[3], 0, 10);
        System.out.println(optimizerName + " on " + train.size() + " training images, " + epochs + " epochs");

        compare(mnistNetwork(), train, test, optimizers, labels, epochs, 64, 0.92);
    }

    private static void regularizeDemo(String[] args){
        System.out.println("== 9. regularization ==");

        int trainLimit = 2000;
        int epochs = 30;
        double l2 = 0.003;
        double dropout = 0.4;
        int patience = 6;
        try{
            if(args.length > 1){
                trainLimit = Integer.parseInt(args[1]);
            }
            if(args.length > 2){
                epochs = Integer.parseInt(args[2]);
            }
            if(args.length > 3){
                l2 = Double.parseDouble(args[3]);
            }
            if(args.length > 4){
                dropout = Double.parseDouble(args[4]);
            }
            if(args.length > 5){
                patience = Integer.parseInt(args[5]);
            }
        }catch(NumberFormatException e){
            System.out.println(USAGE);
            return;
        }
        if(trainLimit < 100 || epochs < 1 || l2 < 0.0 || !(dropout > 0.0 && dropout < 1.0) || patience < 1){
            System.out.println("trainLimit must be at least 100, epochs and patience at least 1, l2 >= 0 and dropout in (0, 1)");
            System.out.println(USAGE);
            return;
        }

        String[] files = findMnistFiles();
        if(files == null){
            return;
        }
        Dataset all = DatasetLoader.fromIdx(files[0], files[1], trainLimit, 10);
        Dataset test = DatasetLoader.fromIdx(files[2], files[3], 0, 10);

        Dataset[] parts = all.stratifiedSplit(0.15, new Random(5));
        Dataset train = parts[0];
        Dataset validation = parts[1];
        System.out.println(train.size() + " training, " + validation.size() + " validation and " + test.size() + " test images, up to " + epochs + " epochs");

        Supplier<NeuralNetwork> network = () -> {
            Random rng = new Random(2026);
            NeuralNetwork net = new NeuralNetwork();
            net.addLayer(new DenseLayer(784, 256, new ReLU(), new HeInitializer(), rng));
            net.addLayer(new DenseLayer(256, 10, new Softmax(), new XavierInitializer(), rng));
            return net;
        };
        Supplier<Optimizer> optimizer = () -> OptimizerFactory.create("adam");

        List<RegularizationComparison.Setup> setups = new ArrayList<>();
        setups.add(new RegularizationComparison.Setup("none", 0.0, 0.0, 0));
        setups.add(new RegularizationComparison.Setup("l2", l2, 0.0, 0));
        setups.add(new RegularizationComparison.Setup("dropout", 0.0, dropout, 0));
        setups.add(new RegularizationComparison.Setup("early", 0.0, 0.0, patience));
        setups.add(new RegularizationComparison.Setup("l2+dropout", l2, dropout, 0));
        setups.add(new RegularizationComparison.Setup("all three", l2, dropout, patience));

        List<RegularizationComparison.Result> results = RegularizationComparison.run(network, optimizer, train, validation, test, setups, epochs, 64, 2026);
        System.out.println();
        System.out.print(RegularizationComparison.formatSetups(results));
        System.out.println();
        System.out.print(RegularizationComparison.formatTable(results));
        System.out.println();
        System.out.println("validation loss while training");
        System.out.print(RegularizationComparison.formatValidationLoss(results, Math.max(1, epochs / 10)));
        System.out.println();
    }

    private static void dataDemo(String[] args){
        System.out.println("== 10. data handling ==");

        String path = args.length > 1 ? args[1] : "data/messy_customers.csv";
        String target = args.length > 2 ? args[2] : null;
        int epochs = 80;
        try{
            if(args.length > 3){
                epochs = Integer.parseInt(args[3]);
            }
        }catch(NumberFormatException e){
            System.out.println(USAGE);
            return;
        }
        if(epochs < 1){
            System.out.println("epochs must be at least 1");
            return;
        }

        PreparedData data;
        try{
            DataOptions options = new DataOptions().missingIndicators(true);
            if(target == null){
                data = DataPipeline.loadByIndex(path, true, -1, options);
            }else{
                data = DataPipeline.loadByName(path, target, options);
            }
        }catch(DataFormatException e){
            System.out.println("the data cannot be used: " + e.getMessage());
            return;
        }

        System.out.println("columns (measured on the rows that have a label):");
        for(ColumnProfile profile : data.getProfiles()){
            System.out.println("  " + profile.describe());
        }
        System.out.println();
        System.out.println("what the pipeline did:");
        System.out.print(data.getReport().format());
        System.out.println();
        System.out.print(data.describe());
        System.out.println("features: " + String.join(", ", data.getPreprocessor().getFeatureNames()));
        System.out.println();

        Dataset train = data.getTrain();
        Dataset test = data.getTest();

        int hidden = Math.max(8, Math.min(64, 2 * train.featureCount()));
        Random rng = new Random(2026);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(train.featureCount(), hidden, new ReLU(), new HeInitializer(), rng));
        net.addLayer(new DenseLayer(hidden, train.classCount(), new Softmax(), new XavierInitializer(), rng));
        System.out.print(net.summary());

        Trainer trainer = new Trainer(net, new CrossEntropy(), OptimizerFactory.create("adam"), rng);
        trainer.train(train, test, epochs, 32, Math.max(1, epochs / 8));

        Evaluation result = trainer.evaluate(test);
        ConfusionMatrix matrix = ConfusionMatrix.of(test.getLabels(), result.getPredictions(), test.classCount());
        System.out.print(matrix.format(test.getClassNames()));
        System.out.print(matrix.report(test.getClassNames()));

        int[] counts = test.classCounts();
        int biggest = 0;
        for(int count : counts){
            biggest = Math.max(biggest, count);
        }
        System.out.println(String.format(Locale.US, "test accuracy %.2f%%, always answering the most common class would give %.2f%%",
            100 * result.getAccuracy(), 100.0 * biggest / test.size()));
        System.out.println();
    }

    private static void compare(Supplier<NeuralNetwork> network, Dataset train, Dataset test, List<Optimizer> optimizers,
                                String[] labels, int epochs, int batchSize, double target){
        List<OptimizerComparison.Result> results = OptimizerComparison.run(network, train, test, optimizers, labels, epochs, batchSize, 2026);
        System.out.println();
        System.out.print(OptimizerComparison.formatTable(results, target));
        System.out.println();
        System.out.println("training loss after every epoch");
        System.out.print(OptimizerComparison.formatLossCurves(results));
        System.out.println();
    }

    private static Supplier<NeuralNetwork> mnistNetwork(){
        return () -> {
            Random rng = new Random(2026);
            NeuralNetwork net = new NeuralNetwork();
            net.addLayer(new DenseLayer(784, 128, new ReLU(), new HeInitializer(), rng));
            net.addLayer(new DenseLayer(128, 10, new Softmax(), new XavierInitializer(), rng));
            return net;
        };
    }

    private static void evalDemo(String[] args){
        System.out.println("== 8. evaluate a saved model ==");

        String modelPath = args.length > 1 ? args[1] : MODEL_PATH;
        if(!new File(modelPath).exists()){
            System.out.println("model file " + modelPath + " not found, create it with: Main mnist");
            System.out.println();
            return;
        }

        String[] files = findMnistFiles();
        if(files == null){
            return;
        }

        NeuralNetwork net = ModelIO.load(modelPath);
        System.out.print(net.summary());
        Dataset test = DatasetLoader.fromIdx(files[2], files[3], 0, 10);

        Trainer trainer = new Trainer(net, new CrossEntropy(), new SGD(0.1), new Random(1));
        Evaluation result = trainer.evaluate(test);
        System.out.println(String.format(Locale.US, "test loss %.4f   accuracy %.2f%%", result.getLoss(), 100 * result.getAccuracy()));

        double top3 = Metrics.topKAccuracy(net.forward(test.getFeatures()), test.getLabels(), 3);
        System.out.println(String.format(Locale.US, "top-3 accuracy %.2f%%", 100 * top3));

        ConfusionMatrix matrix = ConfusionMatrix.of(test.getLabels(), result.getPredictions(), test.classCount());
        System.out.print(matrix.format(test.getClassNames()));
        System.out.print(matrix.report(test.getClassNames()));
        System.out.print(matrix.formatMostConfused(test.getClassNames(), 5));
        System.out.println();
    }

    private static void guiDemo(String[] args){
        String modelPath = args.length > 1 ? args[1] : MODEL_PATH;
        if(!new File(modelPath).exists()){
            System.out.println("model file " + modelPath + " not found, create it with: Main mnist");
            return;
        }
        DigitApp.launch(modelPath);
    }

    private static String[] findMnistFiles(){
        String[] files = {
            findIdxFile(MNIST_FOLDER + "train-images-idx3-ubyte"),
            findIdxFile(MNIST_FOLDER + "train-labels-idx1-ubyte"),
            findIdxFile(MNIST_FOLDER + "t10k-images-idx3-ubyte"),
            findIdxFile(MNIST_FOLDER + "t10k-labels-idx1-ubyte")
        };
        for(String file : files){
            if(file == null){
                System.out.println("MNIST files not found in " + MNIST_FOLDER);
                System.out.println("see " + MNIST_FOLDER + "README.txt for the names of the four files");
                System.out.println();
                return null;
            }
        }
        return files;
    }

    private static String findIdxFile(String basePath){
        if(new File(basePath).exists()){
            return basePath;
        }
        if(new File(basePath + ".gz").exists()){
            return basePath + ".gz";
        }
        return null;
    }
}
