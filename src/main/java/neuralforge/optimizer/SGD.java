package neuralforge.optimizer;

import neuralforge.math.Matrix;
import neuralforge.schedule.ConstantRate;
import neuralforge.schedule.LearningRateSchedule;

public class SGD extends BaseOptimizer{

    public SGD(double learningRate){
        this(new ConstantRate(learningRate));
    }

    public SGD(LearningRateSchedule schedule){
        super(schedule);
    }

    @Override
    protected int bufferCount(){
        return 0;
    }

    @Override
    protected void applyUpdate(Matrix param, Matrix grad, Matrix[] buffers, int step, double rate){
        param.addScaledInPlace(grad, -rate);
    }

    public double getLearningRate(){
        return learningRate();
    }

    @Override
    public String name(){
        return "sgd(" + rateText() + ")";
    }
}
