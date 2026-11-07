package neuralforge.optimizer;

import neuralforge.layer.DenseLayer;
import neuralforge.math.Matrix;
import neuralforge.schedule.ConstantRate;
import neuralforge.schedule.LearningRateSchedule;
import java.util.IdentityHashMap;
import java.util.Map;

public abstract class BaseOptimizer implements Optimizer{

    private static class ParameterState{

        final Matrix[] buffers;

        int step = 0;

        ParameterState(int rows, int cols, int bufferCount){
            buffers = new Matrix[bufferCount];
            for(int i = 0; i < bufferCount; i++){
                buffers[i] = new Matrix(rows, cols);
            }
        }
    }

    private final LearningRateSchedule schedule;
    private double currentRate;

    private final Map<Matrix, ParameterState> states = new IdentityHashMap<>();

    protected BaseOptimizer(LearningRateSchedule schedule){
        if(schedule == null){
            throw new IllegalArgumentException("a learning rate schedule is needed");
        }
        this.schedule = schedule;
        this.currentRate = checkedRate(schedule.rateAt(0), 0);
    }

    protected abstract int bufferCount();

    protected abstract void applyUpdate(Matrix param, Matrix grad, Matrix[] buffers, int step, double rate);

    @Override
    public void startEpoch(int epoch){
        currentRate = checkedRate(schedule.rateAt(epoch), epoch);
    }

    @Override
    public double learningRate(){
        return currentRate;
    }

    @Override
    public void update(DenseLayer layer){
        if(layer.getGradWeights() == null || layer.getGradBiases() == null){
            throw new IllegalStateException("no gradients yet, call backward() before update()");
        }
        updateParameter(layer.getWeights(), layer.getGradWeights());
        updateParameter(layer.getBiases(), layer.getGradBiases());
    }

    public void reset(){
        states.clear();
    }

    private void updateParameter(Matrix param, Matrix grad){
        if(param.rows() != grad.rows() || param.cols() != grad.cols()){
            throw new IllegalArgumentException("gradient is " + grad.shape() + " but the parameter is " + param.shape());
        }

        ParameterState state = states.get(param);
        if(state == null){
            state = new ParameterState(param.rows(), param.cols(), bufferCount());
            states.put(param, state);
        }
        state.step++;
        applyUpdate(param, grad, state.buffers, state.step, currentRate);
    }

    protected String rateText(){
        String text = "lr=" + schedule.rateAt(0);
        if(!(schedule instanceof ConstantRate)){
            text += ", " + schedule.describe();
        }
        return text;
    }

    private static double checkedRate(double rate, int epoch){
        if(Double.isNaN(rate) || Double.isInfinite(rate) || rate < 0){
            throw new IllegalArgumentException("the schedule gave an invalid learning rate " + rate + " for epoch " + epoch);
        }
        return rate;
    }
}
