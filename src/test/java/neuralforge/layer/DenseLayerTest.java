package neuralforge.layer;

import neuralforge.activation.Linear;
import neuralforge.activation.ReLU;
import neuralforge.activation.Sigmoid;
import neuralforge.activation.Softmax;
import neuralforge.init.HeInitializer;
import neuralforge.math.Matrix;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class DenseLayerTest{

    private static final double TOL = 1e-9;

    public static void main(String[] args){
        testForwardLinear();
        testForwardReLU();
        testForwardSigmoidAndSoftmax();
        testInitializedLayer();
        testSavedValues();
        testErrors();
        finish();
    }

    private static Matrix sampleInput(){
        return new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
    }

    private static Matrix sampleWeights(){
        return new Matrix(new double[][]{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}});
    }

    private static void testForwardLinear(){
        Matrix biases = new Matrix(new double[][]{{0.5, -0.5}});
        DenseLayer layer = new DenseLayer(sampleWeights(), biases, new Linear());

        Matrix expected = new Matrix(new double[][]{{2.7, 2.3}, {5.4, 5.9}});
        check("linear layer output", layer.forward(sampleInput()).approxEquals(expected, TOL));
    }

    private static void testForwardReLU(){
        Matrix biases = new Matrix(new double[][]{{-3, -3}});
        DenseLayer layer = new DenseLayer(sampleWeights(), biases, new ReLU());

        Matrix expected = new Matrix(new double[][]{{0, 0}, {1.9, 3.4}});
        check("relu layer output", layer.forward(sampleInput()).approxEquals(expected, TOL));
    }

    private static void testForwardSigmoidAndSoftmax(){
        Matrix biases = Matrix.zeros(1, 2);

        DenseLayer sig = new DenseLayer(sampleWeights(), biases, new Sigmoid());
        Matrix s = sig.forward(sampleInput());
        boolean inRange = true;
        for(int i = 0; i < s.rows(); i++){
            for(int j = 0; j < s.cols(); j++){
                if(s.get(i, j) <= 0 || s.get(i, j) >= 1){
                    inRange = false;
                }
            }
        }
        check("sigmoid layer output in (0, 1)", inRange);

        DenseLayer soft = new DenseLayer(sampleWeights(), biases, new Softmax());
        Matrix p = soft.forward(sampleInput());
        check("softmax layer row 0 sums to 1", near(p.get(0, 0) + p.get(0, 1), 1.0, TOL));
        check("softmax layer row 1 sums to 1", near(p.get(1, 0) + p.get(1, 1), 1.0, TOL));
    }

    private static void testInitializedLayer(){
        DenseLayer layer = new DenseLayer(4, 6, new ReLU(), new HeInitializer(), new Random(1));

        check("input size", layer.inputSize() == 4);
        check("output size", layer.outputSize() == 6);

        check("parameter count", layer.parameterCount() == 30);

        check("biases start at zero", layer.getBiases().approxEquals(Matrix.zeros(1, 6), 0.0));

        Matrix batch = Matrix.randomUniform(5, 4, -1, 1, new Random(2));
        check("output shape", layer.forward(batch).shape().equals("5x6"));

        check("toString", layer.toString().equals("Dense(4 -> 6, relu)"));
    }

    private static void testSavedValues(){
        DenseLayer layer = new DenseLayer(sampleWeights(), Matrix.zeros(1, 2), new Linear());

        check("nothing saved before forward", layer.getLastInput() == null);

        Matrix input = sampleInput();
        Matrix output = layer.forward(input);

        check("saved input", layer.getLastInput() == input);
        check("saved z shape", layer.getLastZ().shape().equals("2x2"));
        check("saved output", layer.getLastOutput() == output);
    }

    private static void testErrors(){
        DenseLayer layer = new DenseLayer(sampleWeights(), Matrix.zeros(1, 2), new Linear());

        expectThrows("wrong input width", () -> layer.forward(Matrix.zeros(4, 2)));

        expectThrows("bias too wide", () -> new DenseLayer(sampleWeights(), Matrix.zeros(1, 3), new Linear()));
        expectThrows("bias not a row", () -> new DenseLayer(sampleWeights(), Matrix.zeros(2, 2), new Linear()));
    }
}
