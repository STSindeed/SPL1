package neuralforge.optimizer;

import neuralforge.activation.Linear;
import neuralforge.layer.DenseLayer;
import neuralforge.math.Matrix;
import static neuralforge.TestHelper.*;

public class OptimizerTest{

    private static final double TOL = 1e-9;

    public static void main(String[] args){
        testUpdateRule();
        testSmallStep();
        testErrors();
        finish();
    }

    private static DenseLayer layerWithGradients(){
        Matrix weights = new Matrix(new double[][]{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}});
        DenseLayer layer = new DenseLayer(weights, Matrix.zeros(1, 2), new Linear());
        layer.forward(new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}}));
        layer.backward(Matrix.filled(2, 2, 1.0));
        return layer;
    }

    private static void testUpdateRule(){
        DenseLayer layer = layerWithGradients();
        new SGD(0.1).update(layer);

        Matrix expectedW = new Matrix(new double[][]{{-0.4, -0.3}, {-0.4, -0.3}, {-0.4, -0.3}});
        check("sgd weights", layer.getWeights().approxEquals(expectedW, TOL));

        check("sgd biases", layer.getBiases().approxEquals(new Matrix(new double[][]{{-0.2, -0.2}}), TOL));
    }

    private static void testSmallStep(){
        DenseLayer layer = layerWithGradients();
        SGD sgd = new SGD(1e-6);
        sgd.update(layer);
        check("small learning rate, small change", near(layer.getWeights().get(0, 0), 0.1, 1e-4));
        check("learning rate getter", sgd.getLearningRate() == 1e-6);
        check("name", sgd.name().startsWith("sgd"));
    }

    private static void testErrors(){
        expectThrows("zero learning rate", () -> new SGD(0));
        expectThrows("negative learning rate", () -> new SGD(-0.5));

        DenseLayer fresh = new DenseLayer(Matrix.zeros(2, 2), Matrix.zeros(1, 2), new Linear());
        expectThrows("update without gradients", () -> new SGD(0.1).update(fresh));
    }
}
