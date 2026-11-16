package neuralforge.experiment;

import neuralforge.data.Dataset;
import neuralforge.loss.CrossEntropy;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.Optimizer;
import neuralforge.training.EarlyStopping;
import neuralforge.training.Evaluation;
import neuralforge.training.Trainer;
import neuralforge.training.TrainingHistory;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.function.Supplier;

public class RegularizationComparison{

    public static class Setup{

        private final String label;
        private final double l2;
        private final double dropout;

        private final int patience;

        public Setup(String label, double l2, double dropout, int patience){
            if(label == null || label.isEmpty()){
                throw new IllegalArgumentException("a setup needs a label");
            }
            if(l2 < 0.0 || Double.isNaN(l2)){
                throw new IllegalArgumentException("L2 strength must be >= 0, got " + l2);
            }
            if(!(dropout >= 0.0 && dropout < 1.0)){
                throw new IllegalArgumentException("dropout rate must be in [0, 1), got " + dropout);
            }
            if(patience < 0){
                throw new IllegalArgumentException("patience cannot be negative, got " + patience);
            }
            this.label = label;
            this.l2 = l2;
            this.dropout = dropout;
            this.patience = patience;
        }

        public String getLabel(){
            return label;
        }

        public double getL2(){
            return l2;
        }

        public double getDropout(){
            return dropout;
        }

        public int getPatience(){
            return patience;
        }

        public String describe(){
            StringBuilder sb = new StringBuilder();
            sb.append(l2 > 0.0 ? "l2=" + l2 : "no l2");
            sb.append(", ");
            sb.append(dropout > 0.0 ? "dropout=" + dropout : "no dropout");
            sb.append(", ");
            sb.append(patience > 0 ? "patience=" + patience : "no early stopping");
            return sb.toString();
        }
    }

    public static class Result{

        private final Setup setup;
        private final TrainingHistory history;
        private final EarlyStopping earlyStopping;
        private final Evaluation trainEvaluation;
        private final Evaluation validationEvaluation;
        private final Evaluation testEvaluation;
        private final int maxEpochs;
        private final double seconds;

        Result(Setup setup, TrainingHistory history, EarlyStopping earlyStopping, Evaluation trainEvaluation,
               Evaluation validationEvaluation, Evaluation testEvaluation, int maxEpochs, double seconds){
            this.setup = setup;
            this.history = history;
            this.earlyStopping = earlyStopping;
            this.trainEvaluation = trainEvaluation;
            this.validationEvaluation = validationEvaluation;
            this.testEvaluation = testEvaluation;
            this.maxEpochs = maxEpochs;
            this.seconds = seconds;
        }

        public Setup getSetup(){
            return setup;
        }

        public TrainingHistory getHistory(){
            return history;
        }

        public EarlyStopping getEarlyStopping(){
            return earlyStopping;
        }

        public int epochsRun(){
            return history.epochCount();
        }

        public int maxEpochs(){
            return maxEpochs;
        }

        public double getSeconds(){
            return seconds;
        }

        public double trainAccuracy(){
            return trainEvaluation.getAccuracy();
        }

        public double validationAccuracy(){
            return validationEvaluation.getAccuracy();
        }

        public double testAccuracy(){
            return testEvaluation.getAccuracy();
        }

        public double testLoss(){
            return testEvaluation.getLoss();
        }

        public double gap(){
            return trainAccuracy() - testAccuracy();
        }
    }

    private RegularizationComparison(){
    }

    public static List<Result> run(Supplier<NeuralNetwork> network, Supplier<Optimizer> optimizer, Dataset train, Dataset validation,
                                   Dataset test, List<Setup> setups, int epochs, int batchSize, long seed){
        if(setups.isEmpty()){
            throw new IllegalArgumentException("nothing to compare");
        }
        if(epochs <= 0 || batchSize <= 0){
            throw new IllegalArgumentException("epochs and batchSize must be positive");
        }

        List<Result> results = new ArrayList<>();
        for(Setup setup : setups){
            NeuralNetwork net = network.get();
            net.setL2(setup.getL2());
            if(setup.getDropout() > 0.0){

                net.enableDropout(setup.getDropout(), new Random(seed + 1));
            }

            Trainer trainer = new Trainer(net, new CrossEntropy(), optimizer.get(), new Random(seed));
            EarlyStopping stopper = null;
            if(setup.getPatience() > 0){
                stopper = new EarlyStopping(EarlyStopping.Monitor.LOSS, setup.getPatience(), 0.0, true);
                trainer.setEarlyStopping(stopper);
            }

            long start = System.nanoTime();
            TrainingHistory history = trainer.train(train, validation, epochs, batchSize, 0);
            double seconds = (System.nanoTime() - start) / 1e9;

            Evaluation onTrain = trainer.evaluate(train);
            Evaluation onValidation = trainer.evaluate(validation);
            Evaluation onTest = trainer.evaluate(test);

            Result result = new Result(setup, history, stopper, onTrain, onValidation, onTest, epochs, seconds);
            results.add(result);
            System.out.println(String.format(Locale.US, "  finished %-12s test accuracy %.2f%%  (%d epochs, %.1fs)",
                setup.getLabel(), 100 * result.testAccuracy(), result.epochsRun(), seconds));
        }
        return results;
    }

    public static String formatTable(List<Result> results){
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-12s %8s %9s %9s %9s %8s %9s %7s\n",
            "setup", "epochs", "train acc", "val acc", "test acc", "gap", "test loss", "time"));

        for(Result r : results){
            String epochs = r.epochsRun() + "/" + r.maxEpochs();
            sb.append(String.format(Locale.US, "%-12s %8s %8.2f%% %8.2f%% %8.2f%% %7.2f%% %9.4f %6.1fs\n",
                r.getSetup().getLabel(), epochs, 100 * r.trainAccuracy(), 100 * r.validationAccuracy(),
                100 * r.testAccuracy(), 100 * r.gap(), r.testLoss(), r.getSeconds()));
        }
        return sb.toString();
    }

    public static String formatSetups(List<Result> results){
        StringBuilder sb = new StringBuilder();
        for(Result r : results){
            sb.append(String.format("%-12s %s", r.getSetup().getLabel(), r.getSetup().describe()));
            EarlyStopping stopper = r.getEarlyStopping();
            if(stopper != null && stopper.hasStopped()){
                sb.append(String.format(Locale.US, "  -> stopped after epoch %d, best epoch %d", stopper.getStoppedEpoch(), stopper.getBestEpoch()));
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public static String formatValidationLoss(List<Result> results, int step){
        if(step < 1){
            throw new IllegalArgumentException("step must be at least 1, got " + step);
        }
        int epochs = 0;
        for(Result r : results){
            epochs = Math.max(epochs, r.epochsRun());
        }

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-6s", "epoch"));
        for(Result r : results){
            sb.append(String.format(" %12s", r.getSetup().getLabel()));
        }
        sb.append("\n");

        for(int e = 0; e < epochs; e++){

            if(e != 0 && (e + 1) % step != 0){
                continue;
            }
            sb.append(String.format("%-6d", e + 1));
            for(Result r : results){
                if(e < r.epochsRun()){
                    sb.append(String.format(Locale.US, " %12.4f", r.getHistory().getValidationLoss(e)));
                }else{
                    sb.append(String.format(" %12s", "-"));
                }
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
