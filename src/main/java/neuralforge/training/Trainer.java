package neuralforge.training;

import neuralforge.data.Dataset;
import neuralforge.data.Shuffler;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.Loss;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import neuralforge.optimizer.Optimizer;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class Trainer{

    private static final int EVALUATION_CHUNK = 1000;

    private final NeuralNetwork network;
    private final Loss loss;
    private final Optimizer optimizer;
    private final Random rng;

    public Trainer(NeuralNetwork network, Loss loss, Optimizer optimizer, Random rng){
        this.network = network;
        this.loss = loss;
        this.optimizer = optimizer;
        this.rng = rng;
    }

    public TrainingHistory train(Dataset trainSet, Dataset validationSet, int epochs, int batchSize, int printEvery){
        if(epochs <= 0 || batchSize <= 0){
            throw new IllegalArgumentException("epochs and batchSize must be positive");
        }
        checkFits(trainSet);
        if(validationSet != null){
            checkFits(validationSet);
        }

        int n = trainSet.size();
        Matrix features = trainSet.getFeatures();
        Matrix targets = trainSet.getTargets();
        int[] labels = trainSet.getLabels();

        List<DenseLayer> layers = network.getLayers();
        DenseLayer lastLayer = layers.get(layers.size() - 1);

        int[] order = Shuffler.range(n);
        TrainingHistory history = new TrainingHistory(epochs);

        for(int epoch = 0; epoch < epochs; epoch++){
            long startTime = System.nanoTime();

            Shuffler.shuffle(order, rng);

            double lossSum = 0.0;
            int correct = 0;

            for(int start = 0; start < n; start += batchSize){
                int end = Math.min(start + batchSize, n);

                Matrix batchFeatures = features.selectRows(order, start, end);
                Matrix batchTargets = targets.selectRows(order, start, end);

                double batchLoss = network.trainBatch(batchFeatures, batchTargets, loss, optimizer);

                lossSum += batchLoss * (end - start);

                int[] guessed = lastLayer.getLastOutput().argmaxRows();
                for(int k = 0; k < guessed.length; k++){
                    if(guessed[k] == labels[order[start + k]]){
                        correct++;
                    }
                }
            }

            double epochLoss = lossSum / n;
            double epochAccuracy = (double) correct / n;

            double validationLoss = Double.NaN;
            double validationAccuracy = Double.NaN;
            if(validationSet != null){
                Evaluation result = evaluate(validationSet);
                validationLoss = result.getLoss();
                validationAccuracy = result.getAccuracy();
            }

            double seconds = (System.nanoTime() - startTime) / 1e9;
            history.record(epochLoss, epochAccuracy, validationLoss, validationAccuracy, seconds);

            boolean lastEpoch = epoch == epochs - 1;
            if(printEvery > 0 && ((epoch + 1) % printEvery == 0 || lastEpoch)){
                printEpoch(epoch + 1, epochs, epochLoss, epochAccuracy, validationLoss, validationAccuracy, seconds);
            }
        }
        return history;
    }

    public Evaluation evaluate(Dataset data){
        checkFits(data);

        int n = data.size();
        Matrix features = data.getFeatures();
        Matrix targets = data.getTargets();
        int[] labels = data.getLabels();

        int[] rows = Shuffler.range(n);
        int[] predictions = new int[n];
        double lossSum = 0.0;
        int correct = 0;

        for(int start = 0; start < n; start += EVALUATION_CHUNK){
            int end = Math.min(start + EVALUATION_CHUNK, n);

            Matrix output = network.forward(features.selectRows(rows, start, end));
            lossSum += loss.compute(output, targets.selectRows(rows, start, end)) * (end - start);

            int[] guessed = output.argmaxRows();
            for(int k = 0; k < guessed.length; k++){
                predictions[start + k] = guessed[k];
                if(guessed[k] == labels[start + k]){
                    correct++;
                }
            }
        }
        return new Evaluation(lossSum / n, (double) correct / n, predictions);
    }

    private void checkFits(Dataset data){
        List<DenseLayer> layers = network.getLayers();
        if(layers.isEmpty()){
            throw new IllegalStateException("the network has no layers");
        }
        int inputs = layers.get(0).inputSize();
        int outputs = layers.get(layers.size() - 1).outputSize();

        if(data.featureCount() != inputs){
            throw new IllegalArgumentException("the data has " + data.featureCount() + " features but the network takes " + inputs);
        }
        if(data.classCount() != outputs){
            throw new IllegalArgumentException("the data has " + data.classCount() + " classes but the network has " + outputs + " outputs");
        }
    }

    private static void printEpoch(int epoch, int epochs, double loss, double accuracy, double validationLoss, double validationAccuracy, double seconds){
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US, "epoch %3d/%d  train loss %.4f acc %6.2f%%", epoch, epochs, loss, 100 * accuracy));
        if(!Double.isNaN(validationLoss)){
            sb.append(String.format(Locale.US, "  |  test loss %.4f acc %6.2f%%", validationLoss, 100 * validationAccuracy));
        }
        sb.append(String.format(Locale.US, "  |  %.1fs", seconds));
        System.out.println(sb);
    }
}
