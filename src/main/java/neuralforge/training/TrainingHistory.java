package neuralforge.training;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

public class TrainingHistory{

    private final double[] trainLoss;
    private final double[] trainAccuracy;
    private final double[] validationLoss;
    private final double[] validationAccuracy;
    private final double[] seconds;
    private int count = 0;

    public TrainingHistory(int maxEpochs){
        if(maxEpochs <= 0){
            throw new IllegalArgumentException("maxEpochs must be positive, got " + maxEpochs);
        }
        trainLoss = new double[maxEpochs];
        trainAccuracy = new double[maxEpochs];
        validationLoss = new double[maxEpochs];
        validationAccuracy = new double[maxEpochs];
        seconds = new double[maxEpochs];
    }

    public void record(double trainLoss, double trainAccuracy, double validationLoss, double validationAccuracy, double seconds){
        if(count == this.trainLoss.length){
            throw new IllegalStateException("history is full");
        }
        this.trainLoss[count] = trainLoss;
        this.trainAccuracy[count] = trainAccuracy;
        this.validationLoss[count] = validationLoss;
        this.validationAccuracy[count] = validationAccuracy;
        this.seconds[count] = seconds;
        count++;
    }

    public int epochCount(){
        return count;
    }

    public double getTrainLoss(int epoch){
        checkEpoch(epoch);
        return trainLoss[epoch];
    }

    public double getTrainAccuracy(int epoch){
        checkEpoch(epoch);
        return trainAccuracy[epoch];
    }

    public double getValidationLoss(int epoch){
        checkEpoch(epoch);
        return validationLoss[epoch];
    }

    public double getValidationAccuracy(int epoch){
        checkEpoch(epoch);
        return validationAccuracy[epoch];
    }

    public double getSeconds(int epoch){
        checkEpoch(epoch);
        return seconds[epoch];
    }

    public int bestValidationEpoch(){
        int best = -1;
        for(int i = 0; i < count; i++){
            if(Double.isNaN(validationAccuracy[i])){
                continue;
            }
            if(best == -1 || validationAccuracy[i] > validationAccuracy[best]){
                best = i;
            }
        }
        return best;
    }

    public void saveCsv(String path){
        Path file = Paths.get(path);
        try{
            Path folder = file.toAbsolutePath().getParent();
            if(folder != null){
                Files.createDirectories(folder);
            }
            try(BufferedWriter out = Files.newBufferedWriter(file, StandardCharsets.UTF_8)){
                out.write("epoch,train_loss,train_accuracy,validation_loss,validation_accuracy,seconds\n");
                for(int i = 0; i < count; i++){
                    out.write(String.format(Locale.US, "%d,%.6f,%.6f,%s,%s,%.3f\n",
                        i + 1, trainLoss[i], trainAccuracy[i],
                        cell(validationLoss[i]), cell(validationAccuracy[i]), seconds[i]));
                }
            }
        }catch(IOException e){
            throw new UncheckedIOException("could not write " + path + ": " + e.getMessage(), e);
        }
    }

    private static String cell(double value){
        if(Double.isNaN(value)){
            return "";
        }
        return String.format(Locale.US, "%.6f", value);
    }

    private void checkEpoch(int epoch){
        if(epoch < 0 || epoch >= count){
            throw new IndexOutOfBoundsException("epoch " + epoch + " is outside 0.." + (count - 1));
        }
    }
}
