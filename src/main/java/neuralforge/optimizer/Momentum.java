package neuralforge.optimizer;

import neuralforge.math.Matrix;
import neuralforge.schedule.ConstantRate;
import neuralforge.schedule.LearningRateSchedule;

public class Momentum extends BaseOptimizer{

    private final double momentum;
    private final boolean nesterov;

    public Momentum(LearningRateSchedule schedule, double momentum, boolean nesterov){
        super(schedule);
        if(!(momentum >= 0 && momentum < 1)){
            throw new IllegalArgumentException("momentum must be in [0, 1), got " + momentum);
        }
        this.momentum = momentum;
        this.nesterov = nesterov;
    }

    public Momentum(double learningRate, double momentum){
        this(new ConstantRate(learningRate), momentum, false);
    }

    @Override
    protected int bufferCount(){
        return 1;
    }

    @Override
    protected void applyUpdate(Matrix param, Matrix grad, Matrix[] buffers, int step, double rate){
        Matrix velocity = buffers[0];
        for(int i = 0; i < param.rows(); i++){
            for(int j = 0; j < param.cols(); j++){
                double g = grad.get(i, j);
                double v = momentum * velocity.get(i, j) + g;
                velocity.set(i, j, v);

                double direction = nesterov ? g + momentum * v : v;
                param.set(i, j, param.get(i, j) - rate * direction);
            }
        }
    }

    @Override
    public String name(){
        return (nesterov ? "nesterov(" : "momentum(") + rateText() + ", mu=" + momentum + ")";
    }
}
