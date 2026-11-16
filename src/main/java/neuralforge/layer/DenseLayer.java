package neuralforge.layer;

import neuralforge.activation.Activation;
import neuralforge.init.Initializer;
import neuralforge.math.Matrix;
import java.util.Locale;
import java.util.Random;

public class DenseLayer{

    private final Matrix weights;
    private final Matrix biases;
    private final Activation activation;

    private Matrix lastInput;
    private Matrix lastZ;

    private Matrix lastActivation;

    private Matrix lastOutput;

    private Matrix gradWeights;
    private Matrix gradBiases;

    private double dropoutRate = 0.0;
    private Random dropoutRng;

    private boolean training = false;

    private Matrix lastMask;

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

        Matrix activated = activation.forward(z);

        Matrix output = activated;
        Matrix mask = null;
        if(training && dropoutRate > 0.0){

            mask = makeDropoutMask(activated.rows(), activated.cols());
            output = activated.hadamard(mask);
        }

        lastInput = input;
        lastZ = z;
        lastActivation = activated;
        lastOutput = output;
        lastMask = mask;
        return output;
    }

    private Matrix makeDropoutMask(int rows, int cols){
        double keepProbability = 1.0 - dropoutRate;
        double scale = 1.0 / keepProbability;

        Matrix mask = new Matrix(rows, cols);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){

                if(dropoutRng.nextDouble() < keepProbability){
                    mask.set(i, j, scale);
                }
            }
        }
        return mask;
    }

    public Matrix backward(Matrix gradOutput){
        if(lastInput == null){
            throw new IllegalStateException("call forward() before backward()");
        }
        if(gradOutput.rows() != lastOutput.rows() || gradOutput.cols() != lastOutput.cols()){
            throw new IllegalArgumentException("gradient " + gradOutput.shape() + " does not match layer output " + lastOutput.shape());
        }

        Matrix gradActivated = gradOutput;
        if(lastMask != null){
            gradActivated = gradOutput.hadamard(lastMask);
        }

        Matrix gradZ = activation.backward(lastZ, lastActivation, gradActivated);

        gradWeights = lastInput.transpose().multiply(gradZ);
        gradBiases = gradZ.columnSums();

        return gradZ.multiply(weights.transpose());
    }

    public void setDropout(double rate, Random rng){
        if(!(rate >= 0.0 && rate < 1.0)){
            throw new IllegalArgumentException("dropout rate must be in [0, 1), got " + rate);
        }
        if(rate > 0.0 && rng == null){
            throw new IllegalArgumentException("dropout needs a random number generator");
        }
        this.dropoutRate = rate;
        this.dropoutRng = rate > 0.0 ? rng : null;
    }

    public double getDropoutRate(){
        return dropoutRate;
    }

    public void setTraining(boolean training){
        this.training = training;
    }

    public boolean isTraining(){
        return training;
    }

    public Matrix getLastMask(){
        return lastMask;
    }

    public void addWeightDecayGradient(double lambda){
        if(gradWeights == null){
            throw new IllegalStateException("call backward() before adding the weight decay gradient");
        }
        gradWeights.addScaledInPlace(weights, lambda);
    }

    public double weightSquaredSum(){
        return weights.sumOfSquares();
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

    public Matrix getGradWeights(){
        return gradWeights;
    }

    public Matrix getGradBiases(){
        return gradBiases;
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
        String text = "Dense(" + inputSize() + " -> " + outputSize() + ", " + activation.name() + ")";
        if(dropoutRate > 0.0){
            text += String.format(Locale.US, " dropout %.2f", dropoutRate);
        }
        return text;
    }
}
