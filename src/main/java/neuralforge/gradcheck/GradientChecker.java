package neuralforge.gradcheck;

import neuralforge.layer.DenseLayer;
import neuralforge.loss.Loss;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import java.util.List;
import java.util.Locale;

public class GradientChecker{

    private static final double H = 1e-5;

    public static double check(NeuralNetwork network, Loss loss, Matrix input, Matrix target, boolean verbose){
        network.computeGradients(input, target, loss);

        double worst = 0.0;
        List<DenseLayer> layers = network.getLayers();
        for(int l = 0; l < layers.size(); l++){
            DenseLayer layer = layers.get(l);

            double weightError = compare(network, loss, input, target, layer.getWeights(), layer.getGradWeights());
            double biasError = compare(network, loss, input, target, layer.getBiases(), layer.getGradBiases());

            if(verbose){
                System.out.println(String.format(Locale.US, "layer %d  weights: %.3e   biases: %.3e", l + 1, weightError, biasError));
            }
            worst = Math.max(worst, Math.max(weightError, biasError));
        }
        return worst;
    }

    private static double compare(NeuralNetwork network, Loss loss, Matrix input, Matrix target, Matrix parameter, Matrix analytical){
        double worst = 0.0;
        for(int i = 0; i < parameter.rows(); i++){
            for(int j = 0; j < parameter.cols(); j++){
                double original = parameter.get(i, j);

                parameter.set(i, j, original + H);
                double lossPlus = network.lossOf(input, target, loss);

                parameter.set(i, j, original - H);
                double lossMinus = network.lossOf(input, target, loss);

                parameter.set(i, j, original);

                double numerical = (lossPlus - lossMinus) / (2.0 * H);
                worst = Math.max(worst, relativeError(analytical.get(i, j), numerical));
            }
        }
        return worst;
    }

    static double relativeError(double a, double b){
        double scale = Math.max(Math.abs(a) + Math.abs(b), 1e-8);
        return Math.abs(a - b) / scale;
    }
}
