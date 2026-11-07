package neuralforge.schedule;

public class CosineDecay implements LearningRateSchedule{

    private final double initial;
    private final double minimum;
    private final int totalEpochs;

    public CosineDecay(double initial, double minimum, int totalEpochs){
        if(!(initial > 0) || Double.isInfinite(initial)){
            throw new IllegalArgumentException("initial rate must be positive, got " + initial);
        }
        if(!(minimum >= 0 && minimum <= initial)){
            throw new IllegalArgumentException("minimum must be between 0 and the initial rate, got " + minimum);
        }
        if(totalEpochs <= 0){
            throw new IllegalArgumentException("totalEpochs must be positive, got " + totalEpochs);
        }
        this.initial = initial;
        this.minimum = minimum;
        this.totalEpochs = totalEpochs;
    }

    @Override
    public double rateAt(int epoch){
        if(epoch < 0){
            throw new IllegalArgumentException("epoch cannot be negative, got " + epoch);
        }
        if(epoch >= totalEpochs){
            return minimum;
        }
        double progress = (double) epoch / totalEpochs;
        return minimum + 0.5 * (initial - minimum) * (1.0 + Math.cos(Math.PI * progress));
    }

    @Override
    public String describe(){
        return "cosine(" + totalEpochs + " epochs)";
    }
}
