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

    public double computeGradients(Matrix input, Matrix target, Loss loss){
        Matrix predicted = forward(input);
        double value = loss.compute(predicted, target);

        Matrix gradient = loss.gradient(predicted, target);
        for(int i = layers.size() - 1; i >= 0; i--){
            gradient = layers.get(i).backward(gradient);
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
        return sb.toString();
    }
}
