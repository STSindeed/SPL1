package neuralforge.optimizer;

import neuralforge.layer.DenseLayer;

public interface Optimizer{

    void update(DenseLayer layer);

    String name();

    default void startEpoch(int epoch){
    }

    default double learningRate(){
        return Double.NaN;
    }
}
