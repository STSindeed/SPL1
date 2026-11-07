package neuralforge.experiment;

import neuralforge.data.Dataset;
import neuralforge.loss.CrossEntropy;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.Optimizer;
import neuralforge.schedule.LearningRateSchedule;
import neuralforge.training.Trainer;
import neuralforge.training.TrainingHistory;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.function.Supplier;

public class OptimizerComparison{

    public static class Result{

        private final String label;
        private final String description;
        private final TrainingHistory history;
        private final double seconds;

        Result(String label, String description, TrainingHistory history, double seconds){
            this.label = label;
            this.description = description;
            this.history = history;
            this.seconds = seconds;
        }

        public String getLabel(){
            return label;
        }

        public String getDescription(){
            return description;
        }

        public TrainingHistory getHistory(){
            return history;
        }

        public double getSeconds(){
            return seconds;
        }

        public double finalTrainLoss(){
            return history.getTrainLoss(history.epochCount() - 1);
        }

        public double finalAccuracy(){
            return history.getValidationAccuracy(history.epochCount() - 1);
        }

        public double bestAccuracy(){
            return history.getValidationAccuracy(history.bestValidationEpoch());
        }
    }

    private OptimizerComparison(){
    }

    public static List<Result> run(Supplier<NeuralNetwork> network, Dataset train, Dataset test,
                                   List<Optimizer> optimizers, String[] labels, int epochs, int batchSize, long seed){
        if(optimizers.size() != labels.length){
            throw new IllegalArgumentException("got " + optimizers.size() + " optimizers but " + labels.length + " labels");
        }
        if(optimizers.isEmpty()){
            throw new IllegalArgumentException("nothing to compare");
        }

        List<Result> results = new ArrayList<>();
        for(int i = 0; i < optimizers.size(); i++){
            Optimizer optimizer = optimizers.get(i);
            NeuralNetwork net = network.get();
            Trainer trainer = new Trainer(net, new CrossEntropy(), optimizer, new Random(seed));

            long start = System.nanoTime();
            TrainingHistory history = trainer.train(train, test, epochs, batchSize, 0);
            double seconds = (System.nanoTime() - start) / 1e9;

            results.add(new Result(labels[i], optimizer.name(), history, seconds));
            System.out.println(String.format(Locale.US, "  finished %-10s test accuracy %.2f%%  (%.1fs)",
                labels[i], 100 * history.getValidationAccuracy(epochs - 1), seconds));
        }
        return results;
    }

    public static String formatTable(List<Result> results, double target){
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US, "%-10s %10s %10s %10s %8s %8s\n",
            "optimizer", "train loss", "final acc", "best acc", String.format(Locale.US, "to %.0f%%", 100 * target), "time"));

        for(Result r : results){
            int reached = r.getHistory().firstEpochReaching(target);
            String reachedText = reached == -1 ? "-" : "epoch " + reached;
            sb.append(String.format(Locale.US, "%-10s %10.4f %9.2f%% %9.2f%% %8s %7.1fs\n",
                r.getLabel(), r.finalTrainLoss(), 100 * r.finalAccuracy(), 100 * r.bestAccuracy(), reachedText, r.getSeconds()));
        }
        return sb.toString();
    }

    public static String formatLossCurves(List<Result> results){
        int epochs = results.get(0).getHistory().epochCount();

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-6s", "epoch"));
        for(Result r : results){
            sb.append(String.format(" %10s", r.getLabel()));
        }
        sb.append("\n");

        for(int e = 0; e < epochs; e++){
            sb.append(String.format("%-6d", e + 1));
            for(Result r : results){
                sb.append(String.format(Locale.US, " %10.4f", r.getHistory().getTrainLoss(e)));
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public static String formatRates(String[] labels, LearningRateSchedule[] schedules, int epochs){
        if(labels.length != schedules.length){
            throw new IllegalArgumentException("got " + schedules.length + " schedules but " + labels.length + " labels");
        }
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-6s", "epoch"));
        for(String label : labels){
            sb.append(String.format(" %10s", label));
        }
        sb.append("\n");

        for(int e = 0; e < epochs; e++){
            sb.append(String.format("%-6d", e + 1));
            for(LearningRateSchedule schedule : schedules){
                sb.append(String.format(Locale.US, " %10.6f", schedule.rateAt(e)));
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
