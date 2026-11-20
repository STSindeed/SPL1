package neuralforge.cli;

import neuralforge.activation.Activation;
import neuralforge.activation.ActivationFactory;
import neuralforge.activation.Softmax;
import neuralforge.data.DataFormatException;
import neuralforge.data.DataPipeline;
import neuralforge.data.Dataset;
import neuralforge.data.PreparedData;
import neuralforge.init.HeInitializer;
import neuralforge.init.Initializer;
import neuralforge.init.XavierInitializer;
import neuralforge.io.ModelBundle;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.CrossEntropy;
import neuralforge.metrics.ConfusionMatrix;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.Optimizer;
import neuralforge.optimizer.OptimizerFactory;
import neuralforge.schedule.LearningRateSchedule;
import neuralforge.schedule.ScheduleFactory;
import neuralforge.training.EarlyStopping;
import neuralforge.training.Evaluation;
import neuralforge.training.Trainer;
import neuralforge.training.TrainingHistory;
import java.io.PrintStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

public class TrainingRun{

    public static class Result{

        private final PreparedData data;
        private final NeuralNetwork network;
        private final TrainingHistory history;
        private final ConfusionMatrix matrix;
        private final double testAccuracy;
        private final double testLoss;
        private final double baselineAccuracy;
        private final int bestEpoch;
        private final boolean stoppedEarly;
        private final String modelPath;

        Result(PreparedData data, NeuralNetwork network, TrainingHistory history, ConfusionMatrix matrix, double testAccuracy, double testLoss,
               double baselineAccuracy, int bestEpoch, boolean stoppedEarly, String modelPath){
            this.data = data;
            this.network = network;
            this.history = history;
            this.matrix = matrix;
            this.testAccuracy = testAccuracy;
            this.testLoss = testLoss;
            this.baselineAccuracy = baselineAccuracy;
            this.bestEpoch = bestEpoch;
            this.stoppedEarly = stoppedEarly;
            this.modelPath = modelPath;
        }

        public PreparedData getData(){
            return data;
        }

        public NeuralNetwork getNetwork(){
            return network;
        }

        public TrainingHistory getHistory(){
            return history;
        }

        public ConfusionMatrix getMatrix(){
            return matrix;
        }

        public double getTestAccuracy(){
            return testAccuracy;
        }

        public double getTestLoss(){
            return testLoss;
        }

        public double getBaselineAccuracy(){
            return baselineAccuracy;
        }

        public int getEpochsRun(){
            return history.epochCount();
        }

        public int getBestEpoch(){
            return bestEpoch;
        }

        public boolean stoppedEarly(){
            return stoppedEarly;
        }

        public String getModelPath(){
            return modelPath;
        }
    }

    private TrainingRun(){
    }

    public static Result run(RunConfig config, PrintStream out){
        PrintStream previous = System.out;
        System.setOut(out);
        try{
            return execute(config, out);
        }finally{
            System.setOut(previous);
        }
    }

    private static Result execute(RunConfig config, PrintStream out){
        out.println("settings:");
        out.print(config.describe());
        out.println();

        PreparedData data;
        if(config.getTarget() == null){
            data = DataPipeline.loadByIndex(config.getDataFile(), true, -1, config.toDataOptions());
        }else{
            data = DataPipeline.loadByName(config.getDataFile(), config.getTarget(), config.toDataOptions());
        }
        out.println("data preparation:");
        out.print(data.getReport().format());
        out.println();
        out.print(data.describe());
        out.println();

        Dataset train = data.getTrain();
        Dataset test = data.getTest();

        NeuralNetwork network = buildNetwork(config, train.featureCount(), train.classCount());
        out.print(network.summary());
        out.println();

        LearningRateSchedule schedule = ScheduleFactory.create(config.getSchedule(), config.effectiveLearningRate(), config.getEpochs());
        Optimizer optimizer = OptimizerFactory.create(config.getOptimizer(), schedule);
        Trainer trainer = new Trainer(network, new CrossEntropy(), optimizer, new Random(config.getTrainSeed()));

        Dataset fitSet = train;
        Dataset watchSet = test;
        EarlyStopping stopper = null;
        if(config.getPatience() > 0){
            Dataset[] split;
            try{
                split = train.stratifiedSplit(config.getValidationFraction(), new Random(config.getTrainSeed()));
            }catch(IllegalArgumentException e){
                throw new DataFormatException("the training rows are too few to hold back a validation part for early stopping: " + e.getMessage());
            }
            fitSet = split[0];
            watchSet = split[1];
            stopper = new EarlyStopping(config.getPatience());
            trainer.setEarlyStopping(stopper);
            out.println("early stopping watches " + watchSet.size() + " rows held back from the " + train.size() + " training rows");
        }

        TrainingHistory history = trainer.train(fitSet, watchSet, config.getEpochs(), config.getBatchSize(), config.getPrintEvery());
        out.println();

        if(stopper != null){
            if(stopper.hasStopped()){
                out.println("early stopping ended the training after epoch " + stopper.getStoppedEpoch() + ", the weights of epoch "
                    + stopper.getBestEpoch() + " were kept");
            }else{
                out.println("early stopping did not trigger, all " + history.epochCount() + " epochs were run, the weights of epoch "
                    + stopper.getBestEpoch() + " were kept");
            }
        }

        Evaluation evaluation = trainer.evaluate(test);
        ConfusionMatrix matrix = ConfusionMatrix.of(test.getLabels(), evaluation.getPredictions(), test.classCount());
        int biggest = 0;
        for(int count : test.classCounts()){
            biggest = Math.max(biggest, count);
        }
        double baseline = (double) biggest / test.size();

        out.print(matrix.format(test.getClassNames()));
        out.print(matrix.report(test.getClassNames()));
        out.println(String.format(Locale.US, "test accuracy %.2f%%, always answering the most common class would give %.2f%%",
            100 * evaluation.getAccuracy(), 100 * baseline));

        int bestEpoch = stopper != null ? stopper.getBestEpoch() : history.bestValidationEpoch() + 1;
        boolean stopped = stopper != null && stopper.hasStopped();

        Map<String, String> notes = new LinkedHashMap<>();
        notes.put("trained_on", config.getDataFile());
        notes.put("optimizer", config.getOptimizer() + " " + String.format(Locale.US, "%.6g", config.effectiveLearningRate()) + " " + config.getSchedule());
        notes.put("epochs_run", String.valueOf(history.epochCount()));
        notes.put("test_accuracy", String.format(Locale.US, "%.4f", evaluation.getAccuracy()));
        notes.put("test_loss", String.format(Locale.US, "%.4f", evaluation.getLoss()));
        notes.put("test_rows", String.valueOf(test.size()));
        ModelBundle.of(network, data, notes).save(config.getModelPath());
        out.println("model saved to " + config.getModelPath() + " and " + ModelBundle.preprocessorPath(config.getModelPath()));

        if(config.getHistoryPath() != null){
            history.saveCsv(config.getHistoryPath());
            out.println("history saved to " + config.getHistoryPath());
        }
        return new Result(data, network, history, matrix, evaluation.getAccuracy(), evaluation.getLoss(), baseline, bestEpoch, stopped,
            config.getModelPath());
    }

    private static NeuralNetwork buildNetwork(RunConfig config, int inputs, int classes){
        Random rng = new Random(config.getModelSeed());
        Initializer hiddenInit = chooseInitializer(config);
        NeuralNetwork network = new NeuralNetwork();

        int previous = inputs;
        List<Integer> hidden = config.getHidden();
        for(int size : hidden){
            Activation activation = ActivationFactory.create(config.getActivation(), 0.01);
            network.addLayer(new DenseLayer(previous, size, activation, hiddenInit, rng));
            previous = size;
        }

        network.addLayer(new DenseLayer(previous, classes, new Softmax(), new XavierInitializer(), rng));

        network.setL2(config.getL2());
        if(config.getDropout() > 0.0){
            network.enableDropout(config.getDropout(), new Random(config.getModelSeed() + 1));
        }
        return network;
    }

    private static Initializer chooseInitializer(RunConfig config){
        String init = config.getInit();
        if(init.equals("auto")){
            init = config.getActivation().equals("relu") || config.getActivation().equals("leaky_relu") ? "he" : "xavier";
        }
        return init.equals("he") ? new HeInitializer() : new XavierInitializer();
    }
}
