package neuralforge.gui;

import neuralforge.layer.DenseLayer;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import java.util.List;

public class DigitClassifier{

    private final NeuralNetwork network;
    private final int inputSize;
    private final int classCount;

    public DigitClassifier(NeuralNetwork network){
        List<DenseLayer> layers = network.getLayers();
        if(layers.isEmpty()){
            throw new IllegalArgumentException("the network has no layers");
        }
        this.network = network;
        this.inputSize = layers.get(0).inputSize();
        this.classCount = layers.get(layers.size() - 1).outputSize();
    }

    public int inputSize(){
        return inputSize;
    }

    public int classCount(){
        return classCount;
    }

    public Prediction predict(double[] pixels){
        if(pixels.length != inputSize){
            throw new IllegalArgumentException("the network takes " + inputSize + " values but got " + pixels.length);
        }

        Matrix input = new Matrix(1, inputSize);
        for(int j = 0; j < inputSize; j++){
            input.set(0, j, pixels[j]);
        }

        Matrix output = network.forward(input);

        double[] probabilities = new double[classCount];
        for(int j = 0; j < classCount; j++){
            probabilities[j] = output.get(0, j);
        }
        return new Prediction(probabilities);
    }

    public String describe(){
        StringBuilder sb = new StringBuilder();
        sb.append(inputSize);
        for(DenseLayer layer : network.getLayers()){
            sb.append(" -> ").append(layer.outputSize());
        }
        return sb.toString();
    }
}
