package neuralforge.optimizer;

import neuralforge.activation.Linear;
import neuralforge.layer.DenseLayer;
import neuralforge.math.Matrix;
import neuralforge.schedule.StepDecay;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class OptimizerUpdateRulesTest{

    public static void main(String[] args){
        testSgd();
        testMomentum();
        testNesterov();
        testRmsProp();
        testAdamFirstSteps();
        testAdamIgnoresGradientSize();
        testBiasIsUpdatedToo();
        testStateIsPerLayer();
        testReset();
        testScheduleChangesTheStep();
        testNames();
        testErrors();
        finish();
    }

    private static final Matrix ONE = new Matrix(new double[][]{{1.0}});

    private static DenseLayer layer(){
        return new DenseLayer(new Matrix(1, 1), new Matrix(1, 1), new Linear());
    }

    private static void step(Optimizer optimizer, DenseLayer layer, double gradient){
        layer.forward(ONE);
        layer.backward(new Matrix(new double[][]{{gradient}}));
        optimizer.update(layer);
    }

    private static double weight(DenseLayer layer){
        return layer.getWeights().get(0, 0);
    }

    private static void testSgd(){
        DenseLayer layer = layer();
        SGD sgd = new SGD(0.1);
        step(sgd, layer, 2.0);
        check("sgd: w = 0 - 0.1 * 2", near(weight(layer), -0.2, 1e-12));
        step(sgd, layer, -1.0);
        check("sgd: w = -0.2 + 0.1", near(weight(layer), -0.1, 1e-12));
    }

    private static void testMomentum(){
        DenseLayer layer = layer();
        Momentum momentum = new Momentum(0.1, 0.9);
        step(momentum, layer, 1.0);
        check("momentum step 1", near(weight(layer), -0.1, 1e-12));
        step(momentum, layer, 1.0);
        check("momentum step 2", near(weight(layer), -0.29, 1e-12));
        step(momentum, layer, 1.0);
        check("momentum step 3", near(weight(layer), -0.561, 1e-12));

        DenseLayer a = layer();
        DenseLayer b = layer();
        Momentum zero = new Momentum(0.1, 0.0);
        SGD sgd = new SGD(0.1);
        for(double g : new double[]{1.0, -2.0, 0.5}){
            step(zero, a, g);
            step(sgd, b, g);
        }
        check("momentum 0 equals sgd", near(weight(a), weight(b), 1e-15));
    }

    private static void testNesterov(){
        DenseLayer layer = layer();
        Momentum nesterov = new Momentum(new neuralforge.schedule.ConstantRate(0.1), 0.9, true);
        step(nesterov, layer, 1.0);
        check("nesterov step 1", near(weight(layer), -0.19, 1e-12));
        step(nesterov, layer, 1.0);
        check("nesterov step 2", near(weight(layer), -0.461, 1e-12));
    }

    private static void testRmsProp(){
        DenseLayer layer = layer();
        RMSProp rmsprop = new RMSProp(0.01);
        step(rmsprop, layer, 2.0);
        check("rmsprop step 1", near(weight(layer), -0.01 * 2.0 / Math.sqrt(0.4), 1e-7));
        step(rmsprop, layer, 2.0);
        check("rmsprop step 2", near(weight(layer), -0.01 * 2.0 / Math.sqrt(0.4) - 0.01 * 2.0 / Math.sqrt(0.76), 1e-7));
    }

    private static void testAdamFirstSteps(){
        DenseLayer layer = layer();
        Adam adam = new Adam(0.01);
        step(adam, layer, 5.0);
        check("adam step 1 has length lr", near(weight(layer), -0.01, 1e-8));
        step(adam, layer, 5.0);
        check("adam step 2 has length lr", near(weight(layer), -0.02, 1e-8));
        step(adam, layer, 5.0);
        check("adam step 3 has length lr", near(weight(layer), -0.03, 1e-8));

        DenseLayer negative = layer();
        step(new Adam(0.01), negative, -5.0);
        check("adam moves against the gradient", near(weight(negative), 0.01, 1e-8));
    }

    private static void testAdamIgnoresGradientSize(){
        DenseLayer big = layer();
        DenseLayer small = layer();
        step(new Adam(0.01), big, 1000.0);
        step(new Adam(0.01), small, 0.001);
        check("huge gradient", near(weight(big), -0.01, 1e-6));
        check("tiny gradient", near(weight(small), -0.01, 1e-3));

        DenseLayer sgdBig = layer();
        step(new SGD(0.01), sgdBig, 1000.0);
        check("sgd step depends on the gradient size", near(weight(sgdBig), -10.0, 1e-9));
    }

    private static void testBiasIsUpdatedToo(){
        DenseLayer layer = layer();
        step(new Momentum(0.1, 0.9), layer, 1.0);
        step(new Momentum(0.1, 0.9), layer, 1.0);
        check("bias was changed", layer.getBiases().get(0, 0) != 0.0);

        DenseLayer both = layer();
        Optimizer adam = new Adam(0.01);
        for(int i = 0; i < 4; i++){
            step(adam, both, 3.0);
        }
        check("weight and bias move together", near(weight(both), both.getBiases().get(0, 0), 1e-12));
    }

    private static void testStateIsPerLayer(){
        Momentum momentum = new Momentum(0.1, 0.9);
        DenseLayer a = layer();
        DenseLayer b = layer();
        step(momentum, a, 1.0);
        step(momentum, a, 1.0);
        step(momentum, a, 1.0);
        step(momentum, b, 1.0);
        check("second layer starts with its own velocity", near(weight(b), -0.1, 1e-12));
        check("first layer kept its own", near(weight(a), -0.561, 1e-12));
    }

    private static void testReset(){
        Momentum momentum = new Momentum(0.1, 0.9);
        DenseLayer layer = layer();
        step(momentum, layer, 1.0);
        step(momentum, layer, 1.0);

        momentum.reset();
        DenseLayer fresh = layer();
        step(momentum, fresh, 1.0);
        check("reset forgets the velocity", near(weight(fresh), -0.1, 1e-12));
    }

    private static void testScheduleChangesTheStep(){
        SGD sgd = new SGD(new StepDecay(0.1, 0.5, 2));
        check("rate before the first epoch call", near(sgd.learningRate(), 0.1, 1e-15));

        DenseLayer layer = layer();
        sgd.startEpoch(0);
        step(sgd, layer, 1.0);
        check("epoch 0 step", near(weight(layer), -0.1, 1e-12));

        sgd.startEpoch(2);
        check("rate in epoch 2", near(sgd.learningRate(), 0.05, 1e-15));
        step(sgd, layer, 1.0);
        check("epoch 2 step is smaller", near(weight(layer), -0.15, 1e-12));
        check("getLearningRate", near(sgd.getLearningRate(), 0.05, 1e-15));
    }

    private static void testNames(){
        check("sgd name keeps its old form", new SGD(0.1).name().equals("sgd(lr=0.1)"));
        check("momentum name", new Momentum(0.02, 0.9).name().startsWith("momentum(") && new Momentum(0.02, 0.9).name().contains("mu=0.9"));
        check("nesterov name", new Momentum(new neuralforge.schedule.ConstantRate(0.02), 0.9, true).name().startsWith("nesterov("));
        check("rmsprop name", new RMSProp(0.001).name().startsWith("rmsprop("));
        check("adam name", new Adam(0.001).name().equals("adam(lr=0.001)"));
        check("name shows a schedule", new SGD(new StepDecay(0.1, 0.5, 2)).name().contains("step("));
    }

    private static void testErrors(){
        expectThrows("sgd with rate 0", () -> new SGD(0.0));
        expectThrows("sgd with negative rate", () -> new SGD(-0.1));
        expectThrows("sgd with NaN rate", () -> new SGD(Double.NaN));
        expectThrows("no schedule", () -> new SGD((neuralforge.schedule.LearningRateSchedule) null));
        expectThrows("momentum 1", () -> new Momentum(0.1, 1.0));
        expectThrows("negative momentum", () -> new Momentum(0.1, -0.1));
        expectThrows("rmsprop rho 1", () -> new RMSProp(new neuralforge.schedule.ConstantRate(0.1), 1.0, 1e-8));
        expectThrows("rmsprop epsilon 0", () -> new RMSProp(new neuralforge.schedule.ConstantRate(0.1), 0.9, 0.0));
        expectThrows("adam beta1 1", () -> new Adam(new neuralforge.schedule.ConstantRate(0.1), 1.0, 0.999, 1e-8));
        expectThrows("adam beta2 negative", () -> new Adam(new neuralforge.schedule.ConstantRate(0.1), 0.9, -0.1, 1e-8));
        expectThrows("adam epsilon 0", () -> new Adam(new neuralforge.schedule.ConstantRate(0.1), 0.9, 0.999, 0.0));

        DenseLayer noGradients = new DenseLayer(1, 1, new Linear(), new neuralforge.init.XavierInitializer(), new Random(1));
        expectThrows("adam without gradients", () -> new Adam(0.01).update(noGradients));
        expectThrows("momentum without gradients", () -> new Momentum(0.1, 0.9).update(noGradients));

        SGD bad = new SGD(new neuralforge.schedule.LearningRateSchedule(){
            @Override
            public double rateAt(int epoch){
                return epoch == 0 ? 0.1 : Double.NaN;
            }

            @Override
            public String describe(){
                return "broken";
            }
        });
        expectThrows("schedule gives NaN", () -> bad.startEpoch(1));
    }
}
