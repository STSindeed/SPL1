package neuralforge.optimizer;

import neuralforge.math.Matrix;
import neuralforge.schedule.ConstantRate;
import neuralforge.schedule.LearningRateSchedule;

public class Adam extends BaseOptimizer{

    private final double beta1;
    private final double beta2;
    private final double epsilon;

    public Adam(LearningRateSchedule schedule, double beta1, double beta2, double epsilon){
        super(schedule);
        if(!(beta1 >= 0 && beta1 < 1)){
            throw new IllegalArgumentException("beta1 must be in [0, 1), got " + beta1);
        }
        if(!(beta2 >= 0 && beta2 < 1)){
            throw new IllegalArgumentException("beta2 must be in [0, 1), got " + beta2);
        }
        if(!(epsilon > 0)){
            throw new IllegalArgumentException("epsilon must be positive, got " + epsilon);
        }
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    public Adam(LearningRateSchedule schedule){
        this(schedule, 0.9, 0.999, 1e-8);
    }

    public Adam(double learningRate){
        this(new ConstantRate(learningRate));
    }

    @Override
    protected int bufferCount(){
        return 2;
    }

    @Override
    protected void applyUpdate(Matrix param, Matrix grad, Matrix[] buffers, int step, double rate){
        Matrix m = buffers[0];
        Matrix v = buffers[1];

        double correction1 = 1.0 - Math.pow(beta1, step);
        double correction2 = 1.0 - Math.pow(beta2, step);

        for(int i = 0; i < param.rows(); i++){
            for(int j = 0; j < param.cols(); j++){
                double g = grad.get(i, j);

                double newM = beta1 * m.get(i, j) + (1.0 - beta1) * g;
                double newV = beta2 * v.get(i, j) + (1.0 - beta2) * g * g;
                m.set(i, j, newM);
                v.set(i, j, newV);

                double mHat = newM / correction1;
                double vHat = newV / correction2;
                param.set(i, j, param.get(i, j) - rate * mHat / (Math.sqrt(vHat) + epsilon));
            }
        }
    }

    @Override
    public String name(){
        return "adam(" + rateText() + ")";
    }
}
