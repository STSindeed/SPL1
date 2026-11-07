package neuralforge.schedule;

public class ConstantRate implements LearningRateSchedule{

    private final double rate;

    public ConstantRate(double rate){
        if(!(rate > 0) || Double.isInfinite(rate)){
            throw new IllegalArgumentException("learning rate must be positive, got " + rate);
        }
        this.rate = rate;
    }

    @Override
    public double rateAt(int epoch){
        if(epoch < 0){
            throw new IllegalArgumentException("epoch cannot be negative, got " + epoch);
        }
        return rate;
    }

    @Override
    public String describe(){
        return "constant";
    }
}
