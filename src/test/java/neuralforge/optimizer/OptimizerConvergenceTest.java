package neuralforge.optimizer;

import neuralforge.activation.Linear;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.loss.MeanSquaredError;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import neuralforge.schedule.ConstantRate;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class OptimizerConvergenceTest{

    public static void main(String[] args){
        testEveryOptimizerLearnsALine();
        testAdamHandlesBadlyScaledInputs();
        testDecayingRateStillConverges();
        finish();
    }

    private static Matrix[] lineData(){
        Matrix x = new Matrix(20, 1);
        Matrix y = new Matrix(20, 1);
        for(int i = 0; i < 20; i++){
            double value = -1.0 + 2.0 * i / 19.0;
            x.set(i, 0, value);
            y.set(i, 0, 3.0 * value + 1.0);
        }
        return new Matrix[]{x, y};
    }

    private static NeuralNetwork oneNeuron(){
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(1, 1, new Linear(), new XavierInitializer(), new Random(4)));
        return net;
    }

    private static void testEveryOptimizerLearnsALine(){
        Matrix[] data = lineData();
        Optimizer[] optimizers = {
            new SGD(0.1),
            new Momentum(0.05, 0.9),
            new Momentum(new ConstantRate(0.05), 0.9, true),
            new RMSProp(0.01),
            new Adam(0.05)
        };

        for(Optimizer optimizer : optimizers){
            NeuralNetwork net = oneNeuron();
            double loss = 0.0;
            for(int step = 0; step < 1500; step++){
                loss = net.trainBatch(data[0], data[1], new MeanSquaredError(), optimizer);
            }
            double weight = net.getLayers().get(0).getWeights().get(0, 0);
            double bias = net.getLayers().get(0).getBiases().get(0, 0);
            check(optimizer.name() + " finds the slope", near(weight, 3.0, 0.05));
            check(optimizer.name() + " finds the intercept", near(bias, 1.0, 0.05));
            check(optimizer.name() + " reaches a small loss", loss < 1e-3);
        }
    }

    private static void testAdamHandlesBadlyScaledInputs(){
        Random rng = new Random(9);
        Matrix x = new Matrix(50, 2);
        Matrix y = new Matrix(50, 1);
        for(int i = 0; i < 50; i++){
            double a = rng.nextDouble() * 2 - 1;
            double b = (rng.nextDouble() * 2 - 1) * 100;
            x.set(i, 0, a);
            x.set(i, 1, b);
            y.set(i, 0, 2.0 * a + 0.05 * b);
        }

        double sgdLoss = lossAfter(new SGD(1e-5), x, y, 400);
        double adamLoss = lossAfter(new Adam(0.05), x, y, 400);
        check("adam learns the badly scaled problem", adamLoss < 0.05);
        check("adam is much better than sgd here", adamLoss * 10 < sgdLoss);
    }

    private static double lossAfter(Optimizer optimizer, Matrix x, Matrix y, int steps){
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(2, 1, new Linear(), new XavierInitializer(), new Random(4)));
        double loss = 0.0;
        for(int i = 0; i < steps; i++){
            loss = net.trainBatch(x, y, new MeanSquaredError(), optimizer);
        }
        return loss;
    }

    private static void testDecayingRateStillConverges(){
        Matrix[] data = lineData();
        NeuralNetwork net = oneNeuron();
        neuralforge.schedule.CosineDecay schedule = new neuralforge.schedule.CosineDecay(0.05, 0.0, 10);
        Optimizer adam = new Adam(schedule);

        double loss = 0.0;
        for(int epoch = 0; epoch < 10; epoch++){
            adam.startEpoch(epoch);
            for(int step = 0; step < 100; step++){
                loss = net.trainBatch(data[0], data[1], new MeanSquaredError(), adam);
            }
        }
        check("adam with cosine decay converges", loss < 1e-3);

        check("rate is the one of the last epoch", near(adam.learningRate(), schedule.rateAt(9), 1e-15));
        check("rate went down", adam.learningRate() < 0.05 / 10);
    }
}
