package neuralforge.schedule;

public class InverseTimeDecay implements LearningRateSchedule{

    private final double initial;
    private final double k;

    public InverseTimeDecay(double initial, double k){
        if(!(initial > 0) || Double.isInfinite(initial)){
            throw new IllegalArgumentException("initial rate must be positive, got " + initial);
        }
        if(!(k >= 0) || Double.isInfinite(k)){
            throw new IllegalArgumentException("k cannot be negative, got " + k);
        }
        this.initial = initial;
        this.k = k;
    }

    @Override
    public double rateAt(int epoch){
        if(epoch < 0){
            throw new IllegalArgumentException("epoch cannot be negative, got " + epoch);
        }
        return initial / (1.0 + k * epoch);
    }

    @Override
    public String describe(){
        return "inverse(k=" + k + ")";
    }
}
