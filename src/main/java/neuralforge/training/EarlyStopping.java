package neuralforge.training;

import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import java.util.Locale;

public class EarlyStopping{

    public enum Monitor{
        LOSS,
        ACCURACY
    }

    private final Monitor monitor;

    private final int patience;

    private final double minDelta;
    private final boolean restoreBest;

    private double best;

    private int bestEpoch;

    private int waited;
    private boolean stopped;
    private int stoppedEpoch;

    private Matrix[] bestParameters;

    public EarlyStopping(Monitor monitor, int patience, double minDelta, boolean restoreBest){
        if(monitor == null){
            throw new IllegalArgumentException("choose what to monitor");
        }
        if(patience < 1){
            throw new IllegalArgumentException("patience must be at least 1, got " + patience);
        }
        if(Double.isNaN(minDelta) || minDelta < 0.0){
            throw new IllegalArgumentException("minDelta must be >= 0, got " + minDelta);
        }
        this.monitor = monitor;
        this.patience = patience;
        this.minDelta = minDelta;
        this.restoreBest = restoreBest;
        reset();
    }

    public EarlyStopping(int patience){
        this(Monitor.LOSS, patience, 0.0, true);
    }

    public void reset(){
        best = Double.NaN;
        bestEpoch = 0;
        waited = 0;
        stopped = false;
        stoppedEpoch = 0;
        bestParameters = null;
    }

    public boolean update(int epoch, double validationLoss, double validationAccuracy, NeuralNetwork network){
        double value = monitor == Monitor.LOSS ? validationLoss : validationAccuracy;

        boolean improved;
        if(Double.isNaN(value)){

            improved = false;
        }else if(bestEpoch == 0){

            improved = true;
        }else if(monitor == Monitor.LOSS){
            improved = value < best - minDelta;
        }else{
            improved = value > best + minDelta;
        }

        if(improved){
            best = value;
            bestEpoch = epoch;
            waited = 0;
            if(restoreBest){

                bestParameters = network.snapshotParameters();
            }
        }else{
            waited++;
        }

        if(waited >= patience){
            stopped = true;
            stoppedEpoch = epoch;
        }
        return stopped;
    }

    public boolean restoreBest(NeuralNetwork network){
        if(!restoreBest || bestParameters == null){
            return false;
        }
        network.restoreParameters(bestParameters);
        return true;
    }

    public Monitor getMonitor(){
        return monitor;
    }

    public int getPatience(){
        return patience;
    }

    public double getMinDelta(){
        return minDelta;
    }

    public boolean isRestoringBest(){
        return restoreBest;
    }

    public double getBestValue(){
        return best;
    }

    public int getBestEpoch(){
        return bestEpoch;
    }

    public boolean hasStopped(){
        return stopped;
    }

    public int getStoppedEpoch(){
        return stoppedEpoch;
    }

    public String describe(){
        String what = monitor == Monitor.LOSS ? "validation loss" : "validation accuracy";
        return String.format(Locale.US, "early stopping on %s, patience %d, min delta %s%s",
            what, patience, minDelta, restoreBest ? ", restores best weights" : "");
    }
}
