package neuralforge.schedule;

public class StepDecay implements LearningRateSchedule{

    private final double initial;
    private final double factor;
    private final int every;

    public StepDecay(double initial, double factor, int every){
        if(!(initial > 0) || Double.isInfinite(initial)){
            throw new IllegalArgumentException("initial rate must be positive, got " + initial);
        }
        if(!(factor > 0 && factor < 1)){
            throw new IllegalArgumentException("factor must be between 0 and 1, got " + factor);
        }
        if(every <= 0){
            throw new IllegalArgumentException("every must be positive, got " + every);
        }
        this.initial = initial;
        this.factor = factor;
        this.every = every;
    }

    @Override
    public double rateAt(int epoch){
        if(epoch < 0){
            throw new IllegalArgumentException("epoch cannot be negative, got " + epoch);
        }
        int steps = epoch / every;
        return initial * Math.pow(factor, steps);
    }

    @Override
    public String describe(){
        return "step(x" + factor + " every " + every + ")";
    }
}
