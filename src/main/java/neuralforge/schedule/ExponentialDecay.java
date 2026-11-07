package neuralforge.schedule;

public class ExponentialDecay implements LearningRateSchedule{

    private final double initial;
    private final double gamma;

    public ExponentialDecay(double initial, double gamma){
        if(!(initial > 0) || Double.isInfinite(initial)){
            throw new IllegalArgumentException("initial rate must be positive, got " + initial);
        }
        if(!(gamma > 0 && gamma <= 1)){
            throw new IllegalArgumentException("gamma must be in (0, 1], got " + gamma);
        }
        this.initial = initial;
        this.gamma = gamma;
    }

    @Override
    public double rateAt(int epoch){
        if(epoch < 0){
            throw new IllegalArgumentException("epoch cannot be negative, got " + epoch);
        }
        return initial * Math.pow(gamma, epoch);
    }

    @Override
    public String describe(){
        return "exp(gamma=" + gamma + ")";
    }
}
