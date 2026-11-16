package neuralforge.network;

import neuralforge.layer.DenseLayer;
import neuralforge.loss.Loss;
import neuralforge.math.Matrix;
import neuralforge.optimizer.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class NeuralNetwork{

    private final List<DenseLayer> layers = new ArrayList<>();

    private double l2 = 0.0;

    public NeuralNetwork addLayer(DenseLayer layer){

        if(!layers.isEmpty()){
            int previousOutput = layers.get(layers.size() - 1).outputSize();
            if(layer.inputSize() != previousOutput){
                throw new IllegalArgumentException("layer expects " + layer.inputSize() + " inputs but the previous layer gives " + previousOutput);
            }
        }
        layers.add(layer);
        return this;
    }

    public List<DenseLayer> getLayers(){
        return layers;
    }

    public Matrix forward(Matrix input){
        if(layers.isEmpty()){
            throw new IllegalStateException("the network has no layers");
        }
        Matrix current = input;
        for(DenseLayer layer : layers){
            current = layer.forward(current);
        }
        return current;
    }

    public double lossOf(Matrix input, Matrix target, Loss loss){
        return loss.compute(forward(input), target);
    }

    public double totalLossOf(Matrix input, Matrix target, Loss loss){
        return lossOf(input, target, loss) + regularizationPenalty();
    }

    public int[] predictClasses(Matrix input){
        return forward(input).argmaxRows();
    }

    public double accuracy(Matrix input, Matrix oneHotTarget){
        int[] predicted = predictClasses(input);
        int[] actual = oneHotTarget.argmaxRows();
        int correct = 0;
        for(int i = 0; i < predicted.length; i++){
            if(predicted[i] == actual[i]){
                correct++;
            }
        }
        return (double) correct / predicted.length;
    }

    public NeuralNetwork setL2(double lambda){
        if(Double.isNaN(lambda) || Double.isInfinite(lambda) || lambda < 0.0){
            throw new IllegalArgumentException("L2 strength must be a finite number >= 0, got " + lambda);
        }
        this.l2 = lambda;
        return this;
    }

    public double getL2(){
        return l2;
    }

    public double regularizationPenalty(){
        if(l2 == 0.0){
            return 0.0;
        }
        double sum = 0.0;
        for(DenseLayer layer : layers){
            sum += layer.weightSquaredSum();
        }
        return 0.5 * l2 * sum;
    }

    public NeuralNetwork enableDropout(double rate, Random rng){
        if(layers.size() < 2){
            throw new IllegalStateException("dropout needs at least one hidden layer, the network has " + layers.size() + " layer(s)");
        }
        for(int i = 0; i < layers.size() - 1; i++){
            layers.get(i).setDropout(rate, rng);
        }
        return this;
    }

    public void setTraining(boolean training){
        for(DenseLayer layer : layers){
            layer.setTraining(training);
        }
    }

    public double computeGradients(Matrix input, Matrix target, Loss loss){
        Matrix predicted = forward(input);
        double value = loss.compute(predicted, target);

        Matrix gradient = loss.gradient(predicted, target);
        for(int i = layers.size() - 1; i >= 0; i--){
            gradient = layers.get(i).backward(gradient);
        }

        if(l2 > 0.0){
            for(DenseLayer layer : layers){
                layer.addWeightDecayGradient(l2);
            }
        }
        return value;
    }

    public double trainBatch(Matrix input, Matrix target, Loss loss, Optimizer optimizer){
        double value = computeGradients(input, target, loss);
        for(DenseLayer layer : layers){
            optimizer.update(layer);
        }
        return value;
    }

    public double[] fit(Matrix input, Matrix target, Loss loss, Optimizer optimizer, int epochs, int batchSize, Random rng, int printEvery){
        if(input.rows() != target.rows()){
            throw new IllegalArgumentException("input has " + input.rows() + " samples but target has " + target.rows());
        }
        if(epochs <= 0 || batchSize <= 0){
            throw new IllegalArgumentException("epochs and batchSize must be positive");
        }

        int n = input.rows();

        int[] order = new int[n];
        for(int i = 0; i < n; i++){
            order[i] = i;
        }

        double[] history = new double[epochs];
        setTraining(true);
        try{
            for(int epoch = 0; epoch < epochs; epoch++){
                shuffle(order, rng);

                double total = 0.0;
                for(int start = 0; start < n; start += batchSize){
                    int end = Math.min(start + batchSize, n);
                    Matrix batchInput = input.selectRows(order, start, end);
                    Matrix batchTarget = target.selectRows(order, start, end);

                    double batchLoss = trainBatch(batchInput, batchTarget, loss, optimizer);

                    total += batchLoss * (end - start);
                }
                history[epoch] = total / n;

                if(printEvery > 0 && (epoch + 1) % printEvery == 0){
                    System.out.println(String.format(Locale.US, "epoch %4d  loss %.6f", epoch + 1, history[epoch]));
                }
            }
        }finally{
            setTraining(false);
        }
        return history;
    }

    static void shuffle(int[] array, Random rng){
        for(int i = array.length - 1; i > 0; i--){
            int j = rng.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    public Matrix[] snapshotParameters(){
        Matrix[] copy = new Matrix[layers.size() * 2];
        for(int i = 0; i < layers.size(); i++){
            copy[2 * i] = layers.get(i).getWeights().copy();
            copy[2 * i + 1] = layers.get(i).getBiases().copy();
        }
        return copy;
    }

    public void restoreParameters(Matrix[] snapshot){
        if(snapshot == null || snapshot.length != layers.size() * 2){
            throw new IllegalArgumentException("the snapshot does not belong to a network with " + layers.size() + " layers");
        }
        for(int i = 0; i < layers.size(); i++){
            layers.get(i).getWeights().copyFrom(snapshot[2 * i]);
            layers.get(i).getBiases().copyFrom(snapshot[2 * i + 1]);
        }
    }

    public int parameterCount(){
        int total = 0;
        for(DenseLayer layer : layers){
            total += layer.parameterCount();
        }
        return total;
    }

    public String summary(){
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < layers.size(); i++){
            sb.append("layer ").append(i + 1).append(": ").append(layers.get(i));
            sb.append("  (").append(layers.get(i).parameterCount()).append(" parameters)\n");
        }
        sb.append("total parameters: ").append(parameterCount()).append("\n");
        if(l2 > 0.0){
            sb.append("L2 strength: ").append(l2).append("\n");
        }
        return sb.toString();
    }
}
