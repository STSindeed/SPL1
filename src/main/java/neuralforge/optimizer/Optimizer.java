package neuralforge.optimizer;

import neuralforge.layer.DenseLayer;

public interface Optimizer{
    void update(DenseLayer layer);

    String name();
}
