package neuralforge;

import neuralforge.activation.ReLU;
import neuralforge.activation.Sigmoid;
import neuralforge.activation.Softmax;
import neuralforge.activation.Tanh;
import neuralforge.data.Dataset;
import neuralforge.data.DatasetLoader;
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
import neuralforge.optimizer.SGD;
import neuralforge.preprocess.StandardScaler;
import neuralforge.training.Evaluation;
import neuralforge.training.Trainer;
import neuralforge.training.TrainingHistory;
import java.io.File;
import java.util.Locale;
import java.util.Random;

public class Main{

    private static final String USAGE = "usage: Main [all | check | xor | blobs | iris | mnist [trainLimit [epochs]] | eval [modelFile] | gui [modelFile]]";

    private static final String MODEL_PATH = "models/mnist.nfm";

    private static final String MNIST_FOLDER = "data/mnist/";

    public static void main(String[] args){
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
        if(mode.equals("eval")){
            evalDemo(args);
            known = true;
        }
        if(mode.equals("gui")){
            guiDemo(args);
            known = true;
        }

        if(!known){
            System.out.println("unknown demo '" + mode + "'");
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

        Dataset all = DatasetLoader.fromCsv(path, true, -1);
        System.out.print(all.describe());

        Random rng = new Random(42);

        Dataset[] parts = all.stratifiedSplit(0.2, rng);

        StandardScaler scaler = new StandardScaler();
        Dataset train = parts[0].withFeatures(scaler.fitTransform(parts[0].getFeatures()));
        Dataset test = parts[1].withFeatures(scaler.transform(parts[1].getFeatures()));
        System.out.println("train: " + train.size() + " samples, test: " + test.size() + " samples");

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

    private static void mnistDemo(String[] args){
        System.out.println("== 5. mnist ==");

        int trainLimit = 0;
        int epochs = 5;
        try{
            if(args.length > 1){
                trainLimit = Integer.parseInt(args[1]);
            }
            if(args.length > 2){
                epochs = Integer.parseInt(args[2]);
            }
        }catch(NumberFormatException e){
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
        System.out.print(net.summary());

        Trainer trainer = new Trainer(net, new CrossEntropy(), new SGD(0.1), rng);
        TrainingHistory history = trainer.train(train, test, epochs, 64, 1);
        history.saveCsv("results/mnist_history.csv");

        ModelIO.save(net, MODEL_PATH);
        System.out.println("model saved to " + MODEL_PATH);

        int best = history.bestValidationEpoch();
        System.out.println(String.format(Locale.US, "best test accuracy %.2f%% at epoch %d", 100 * history.getValidationAccuracy(best), best + 1));

        Evaluation result = trainer.evaluate(test);
        ConfusionMatrix matrix = ConfusionMatrix.of(test.getLabels(), result.getPredictions(), test.classCount());
        System.out.print(matrix.format(test.getClassNames()));
        System.out.print(matrix.report(test.getClassNames()));
        System.out.println();
    }

    private static void evalDemo(String[] args){
        System.out.println("== 6. evaluate a saved model ==");

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
