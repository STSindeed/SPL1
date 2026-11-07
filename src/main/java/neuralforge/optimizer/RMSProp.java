package neuralforge.optimizer;

import neuralforge.math.Matrix;
import neuralforge.schedule.ConstantRate;
import neuralforge.schedule.LearningRateSchedule;

public class RMSProp extends BaseOptimizer{

    private final double rho;
    private final double epsilon;

    public RMSProp(LearningRateSchedule schedule, double rho, double epsilon){
        super(schedule);
        if(!(rho >= 0 && rho < 1)){
            throw new IllegalArgumentException("rho must be in [0, 1), got " + rho);
        }
        if(!(epsilon > 0)){
            throw new IllegalArgumentException("epsilon must be positive, got " + epsilon);
        }
        this.rho = rho;
        this.epsilon = epsilon;
    }

    public RMSProp(double learningRate){
        this(new ConstantRate(learningRate), 0.9, 1e-8);
    }

    @Override
    protected int bufferCount(){
        return 1;
    }

    @Override
    protected void applyUpdate(Matrix param, Matrix grad, Matrix[] buffers, int step, double rate){
        Matrix average = buffers[0];
        for(int i = 0; i < param.rows(); i++){
            for(int j = 0; j < param.cols(); j++){
                double g = grad.get(i, j);
                double s = rho * average.get(i, j) + (1.0 - rho) * g * g;
                average.set(i, j, s);
                param.set(i, j, param.get(i, j) - rate * g / (Math.sqrt(s) + epsilon));
            }
        }
    }

    @Override
    public String name(){
        return "rmsprop(" + rateText() + ", rho=" + rho + ")";
    }
}
