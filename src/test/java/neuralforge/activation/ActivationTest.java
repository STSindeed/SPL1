package neuralforge.activation;

import neuralforge.math.Matrix;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class ActivationTest{

    private static final double TOL = 1e-9;

    public static void main(String[] args){
        testSigmoid();
        testTanh();
        testReLU();
        testLeakyReLU();
        testLinear();
        testSoftmax();
        testBackward();
        finish();
    }

    private static Matrix row(double... values){
        return new Matrix(new double[][]{values});
    }

    private static void testSigmoid(){
        Sigmoid s = new Sigmoid();
        Matrix out = s.forward(row(0, 2, -2));

        check("sigmoid(0) = 0.5", near(out.get(0, 0), 0.5, TOL));
        check("sigmoid(2)", near(out.get(0, 1), 0.8807970779778823, TOL));
        check("sigmoid(-2)", near(out.get(0, 2), 0.11920292202211755, TOL));

        check("sigmoid symmetry", near(out.get(0, 2), 1 - out.get(0, 1), TOL));

        Matrix big = s.forward(row(-1000, 1000));
        check("sigmoid(-1000) is 0", big.get(0, 0) == 0.0);
        check("sigmoid(1000) is 1", big.get(0, 1) == 1.0);

        check("sigmoid keeps shape", out.shape().equals("1x3"));
    }

    private static void testTanh(){
        Tanh t = new Tanh();
        Matrix out = t.forward(row(0, 1, -1));

        check("tanh(0) = 0", near(out.get(0, 0), 0.0, TOL));
        check("tanh(1)", near(out.get(0, 1), 0.7615941559557649, TOL));

        check("tanh odd", near(out.get(0, 2), -out.get(0, 1), TOL));
    }

    private static void testReLU(){
        Matrix out = new ReLU().forward(row(-3, 0, 2.5));
        check("relu negative", out.get(0, 0) == 0.0);
        check("relu zero", out.get(0, 1) == 0.0);
        check("relu positive", out.get(0, 2) == 2.5);
    }

    private static void testLeakyReLU(){
        Matrix out = new LeakyReLU(0.1).forward(row(-3, 4));
        check("leaky negative", near(out.get(0, 0), -0.3, TOL));
        check("leaky positive", out.get(0, 1) == 4.0);

        Matrix def = new LeakyReLU().forward(row(-100));
        check("leaky default slope", near(def.get(0, 0), -1.0, TOL));
    }

    private static void testLinear(){
        Matrix in = row(-1.5, 0, 7);
        Matrix out = new Linear().forward(in);
        check("linear returns same values", out.approxEquals(in, TOL));

        out.set(0, 0, 99);
        check("linear output is a copy", in.get(0, 0) == -1.5);
    }

    private static void testSoftmax(){
        Softmax sm = new Softmax();

        Matrix out = sm.forward(row(1, 2, 3));
        check("softmax[0]", near(out.get(0, 0), 0.09003057, 1e-7));
        check("softmax[1]", near(out.get(0, 1), 0.24472847, 1e-7));
        check("softmax[2]", near(out.get(0, 2), 0.66524096, 1e-7));

        Matrix two = sm.forward(new Matrix(new double[][]{{1, 2, 3}, {-5, 0, 5}}));
        for(int i = 0; i < two.rows(); i++){
            double sum = 0;
            for(int j = 0; j < two.cols(); j++){
                sum += two.get(i, j);
            }
            check("softmax row " + i + " sums to 1", near(sum, 1.0, TOL));
        }

        Matrix shifted = sm.forward(row(1001, 1002, 1003));
        check("softmax shift invariance", shifted.approxEquals(out, 1e-9));

        check("softmax no NaN", !Double.isNaN(shifted.get(0, 0)));

        Matrix uniform = sm.forward(row(4, 4, 4, 4));
        check("softmax uniform", near(uniform.get(0, 2), 0.25, TOL));
    }

    private static void testBackward(){
        Activation[] all = {new Sigmoid(), new Tanh(), new ReLU(), new LeakyReLU(0.1), new Linear(), new Softmax()};
        Random rng = new Random(11);

        for(Activation act : all){
            Matrix z = Matrix.randomUniform(3, 4, -2, 2, rng);
            Matrix g = Matrix.randomUniform(3, 4, -1, 1, rng);

            Matrix analytical = act.backward(z, act.forward(z), g);
            check(act.name() + " backward keeps shape", analytical.shape().equals("3x4"));

            double h = 1e-6;
            double worst = 0;
            for(int i = 0; i < z.rows(); i++){
                for(int j = 0; j < z.cols(); j++){
                    double original = z.get(i, j);

                    z.set(i, j, original + h);
                    double plus = weightedSum(act.forward(z), g);
                    z.set(i, j, original - h);
                    double minus = weightedSum(act.forward(z), g);
                    z.set(i, j, original);

                    double numerical = (plus - minus) / (2 * h);
                    worst = Math.max(worst, Math.abs(numerical - analytical.get(i, j)));
                }
            }
            check(act.name() + " backward matches numerical slope", worst < 1e-6);
        }

        Sigmoid sig = new Sigmoid();
        Matrix z0 = row(0);
        check("sigmoid slope at 0", near(sig.backward(z0, sig.forward(z0), row(1)).get(0, 0), 0.25, TOL));

        Matrix zr = row(-1, 2);
        Matrix gr = new ReLU().backward(zr, new ReLU().forward(zr), row(5, 5));
        check("relu blocks negative", gr.get(0, 0) == 0.0);
        check("relu passes positive", gr.get(0, 1) == 5.0);

        Softmax sm = new Softmax();
        Matrix zs = row(0.5, -1, 2);
        Matrix gs = sm.backward(zs, sm.forward(zs), row(1, 2, 3));
        check("softmax gradient row sums to 0", near(gs.get(0, 0) + gs.get(0, 1) + gs.get(0, 2), 0.0, TOL));

        expectThrows("backward bad gradient shape", () -> sm.backward(zs, sm.forward(zs), Matrix.zeros(2, 3)));
        expectThrows("sigmoid backward bad gradient shape", () -> sig.backward(z0, sig.forward(z0), Matrix.zeros(2, 2)));
    }

    private static double weightedSum(Matrix a, Matrix b){
        double sum = 0;
        for(int i = 0; i < a.rows(); i++){
            for(int j = 0; j < a.cols(); j++){
                sum += a.get(i, j) * b.get(i, j);
            }
        }
        return sum;
    }
}
