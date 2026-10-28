package neuralforge.training;

public class Evaluation{

    private final double loss;
    private final double accuracy;
    private final int[] predictions;

    public Evaluation(double loss, double accuracy, int[] predictions){
        this.loss = loss;
        this.accuracy = accuracy;
        this.predictions = predictions;
    }

    public double getLoss(){
        return loss;
    }

    public double getAccuracy(){
        return accuracy;
    }

    public int[] getPredictions(){
        return predictions.clone();
    }
}
