package neuralforge.layer;

import neuralforge.activation.Activation;
import neuralforge.init.Initializer;
import neuralforge.math.Matrix;
import java.util.Random;

public class DenseLayer {

    private final Matrix weights;
    private final Matrix biases;
    private final Activation activation;

    private Matrix lastInput;
    private Matrix lastZ;
    private Matrix lastOutput;

    public DenseLayer(int inputSize, int outputSize, Activation activation, Initializer initializer, Random rng){
        this(initializer.createWeights(inputSize, outputSize, rng), Matrix.zeros(1, outputSize), activation);
    }

    public DenseLayer(Matrix weights, Matrix biases, Activation activation){
        if(biases.rows() != 1 || biases.cols() != weights.cols()){
            throw new IllegalArgumentException("biases must be 1x" + weights.cols() + " but they are " + biases.shape());
        }
        this.weights = weights;
        this.biases = biases;
        this.activation = activation;
    }

    public Matrix forward(Matrix input){
        if(input.cols() != weights.rows()){
            throw new IllegalArgumentException("layer expects " + weights.rows() + " inputs per sample but got " + input.cols());
        }

        Matrix z = input.multiply(weights).addBias(biases);

        Matrix output = activation.forward(z);

        lastInput = input;
        lastZ = z;
        lastOutput = output;
        return output;
    }


    public int inputSize(){
        return weights.rows();
    }

    public int outputSize(){
        return weights.cols();
    }

    public int parameterCount(){
        return weights.rows() * weights.cols() + biases.cols();
    }

    public Matrix getWeights(){
        return weights;
    }

    public Matrix getBiases(){
        return biases;
    }

    public Activation getActivation(){
        return activation;
    }

    public Matrix getLastInput(){
        return lastInput;
    }

    public Matrix getLastZ(){
        return lastZ;
    }

    public Matrix getLastOutput(){
        return lastOutput;
    }

    @Override
    public String toString(){
        return "Dense(" + inputSize() + " -> " + outputSize() + ", " + activation.name() + ")";
    }
}
