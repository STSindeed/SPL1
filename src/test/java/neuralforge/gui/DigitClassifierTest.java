package neuralforge.gui;

import neuralforge.activation.Softmax;
import neuralforge.layer.DenseLayer;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import java.util.Arrays;
import static neuralforge.TestHelper.*;

public class DigitClassifierTest{

    public static void main(String[] args){
        testPrediction();
        testTop();
        testClassifier();
        testErrors();
        finish();
    }

    public static NeuralNetwork pixelNetwork(){
        Matrix weights = new Matrix(16, 10);
        for(int k = 0; k < 10; k++){
            weights.set(k, k, 6.0);
        }
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(weights, new Matrix(1, 10), new Softmax()));
        return net;
    }

    private static void testPrediction(){
        Prediction p = new Prediction(new double[]{0.1, 0.7, 0.2});
        check("digit", p.digit() == 1);
        check("confidence", p.confidence() == 0.7);
        check("probability of a class", p.probability(2) == 0.2);

        check("tie goes to the smaller class", new Prediction(new double[]{0.4, 0.4, 0.2}).digit() == 0);

        double[] copy = p.getProbabilities();
        copy[1] = 0.0;
        check("probabilities are a copy", p.confidence() == 0.7);
        expectThrows("no classes", () -> new Prediction(new double[0]));
    }

    private static void testTop(){
        Prediction p = new Prediction(new double[]{0.05, 0.4, 0.1, 0.3, 0.15});
        check("top 3", Arrays.equals(p.top(3), new int[]{1, 3, 4}));
        check("top 1 is the digit", p.top(1)[0] == p.digit());
        check("top all", Arrays.equals(p.top(5), new int[]{1, 3, 4, 2, 0}));
        check("top with ties", Arrays.equals(new Prediction(new double[]{0.3, 0.3, 0.4}).top(3), new int[]{2, 0, 1}));
        expectThrows("k is 0", () -> p.top(0));
        expectThrows("k too big", () -> p.top(6));
    }

    private static void testClassifier(){
        DigitClassifier classifier = new DigitClassifier(pixelNetwork());
        check("input size", classifier.inputSize() == 16);
        check("class count", classifier.classCount() == 10);
        check("description", classifier.describe().equals("16 -> 10"));

        double[] pixels = new double[16];
        pixels[3] = 1.0;
        Prediction p = classifier.predict(pixels);
        check("brightest pixel decides", p.digit() == 3);
        check("confident", p.confidence() > 0.9);

        double sum = 0.0;
        for(double value : p.getProbabilities()){
            sum += value;
        }
        check("probabilities sum to 1", near(sum, 1.0, 1e-12));

        check("repeatable", classifier.predict(pixels).digit() == 3);

        pixels[3] = 0.0;
        pixels[7] = 0.8;
        check("another pixel", classifier.predict(pixels).digit() == 7);
    }

    private static void testErrors(){
        DigitClassifier classifier = new DigitClassifier(pixelNetwork());
        expectThrows("wrong number of pixels", () -> classifier.predict(new double[15]));
        expectThrows("network without layers", () -> new DigitClassifier(new NeuralNetwork()));
    }
}
