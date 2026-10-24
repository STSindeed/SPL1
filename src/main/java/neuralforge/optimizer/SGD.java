package neuralforge.optimizer;

import neuralforge.layer.DenseLayer;

public class SGD implements Optimizer{

    private final double learningRate;

    public SGD(double learningRate){
        if(learningRate <= 0){
            throw new IllegalArgumentException("learning rate must be positive, got " + learningRate);
        }
        this.learningRate = learningRate;
    }

    @Override
    public void update(DenseLayer layer){
        if(layer.getGradWeights() == null || layer.getGradBiases() == null){
            throw new IllegalStateException("no gradients yet, call backward() before update()");
        }

        layer.getWeights().addScaledInPlace(layer.getGradWeights(), -learningRate);
        layer.getBiases().addScaledInPlace(layer.getGradBiases(), -learningRate);
    }

    public double getLearningRate(){
        return learningRate;
    }

    @Override
    public String name(){
        return "sgd(lr=" + learningRate + ")";
    }
}
