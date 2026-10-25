package neuralforge.loss;

import neuralforge.math.Matrix;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class LossTest{

    private static final double TOL = 1e-9;

    public static void main(String[] args){
        testMeanSquaredError();
        testCrossEntropy();
        testGradients();
        testErrors();
        finish();
    }

    private static void testMeanSquaredError(){
        Loss mse = new MeanSquaredError();

        Matrix p1 = new Matrix(new double[][]{{1, 2}});
        Matrix t1 = new Matrix(new double[][]{{0, 0}});
        check("mse single row", near(mse.compute(p1, t1), 2.5, TOL));

        Matrix p2 = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix t2 = new Matrix(new double[][]{{1, 2}, {3, 6}});
        check("mse two rows", near(mse.compute(p2, t2), 1.0, TOL));

        check("mse perfect", mse.compute(p2, p2) == 0.0);

        check("mse symmetric", near(mse.compute(p2, t2), mse.compute(t2, p2), TOL));
    }

    private static void testCrossEntropy(){
        Loss ce = new CrossEntropy();

        Matrix p1 = new Matrix(new double[][]{{0.7, 0.2, 0.1}});
        Matrix t1 = new Matrix(new double[][]{{1, 0, 0}});
        check("cross entropy single row", near(ce.compute(p1, t1), -Math.log(0.7), TOL));

        Matrix p2 = new Matrix(new double[][]{{0.7, 0.2, 0.1}, {0.1, 0.8, 0.1}});
        Matrix t2 = new Matrix(new double[][]{{1, 0, 0}, {0, 1, 0}});
        double expected = -(Math.log(0.7) + Math.log(0.8)) / 2;
        check("cross entropy batch average", near(ce.compute(p2, t2), expected, TOL));

        Matrix sure = new Matrix(new double[][]{{0, 1, 0}});
        Matrix surelyRight = new Matrix(new double[][]{{0, 1, 0}});
        check("cross entropy perfect", near(ce.compute(sure, surelyRight), 0.0, TOL));

        Matrix zeroProb = new Matrix(new double[][]{{0, 1}});
        Matrix target = new Matrix(new double[][]{{1, 0}});
        double clipped = ce.compute(zeroProb, target);
        check("cross entropy zero probability is finite", !Double.isInfinite(clipped) && !Double.isNaN(clipped));
        check("cross entropy zero probability is large", clipped > 20);

        Matrix guess = Matrix.filled(1, 10, 0.1);
        Matrix digit = new Matrix(new double[][]{{0, 0, 0, 0, 1, 0, 0, 0, 0, 0}});
        check("cross entropy random guess", near(ce.compute(guess, digit), Math.log(10), TOL));

        Matrix better = new Matrix(new double[][]{{0.9, 0.05, 0.05}});
        Matrix worse = new Matrix(new double[][]{{0.2, 0.4, 0.4}});
        check("worse prediction, bigger loss", ce.compute(worse, t1) > ce.compute(better, t1));
    }

    private static void testGradients(){
        Loss mse = new MeanSquaredError();
        Matrix g1 = mse.gradient(new Matrix(new double[][]{{1, 2}}), new Matrix(new double[][]{{0, 0}}));
        check("mse gradient by hand", g1.approxEquals(new Matrix(new double[][]{{1, 2}}), TOL));

        Loss ce = new CrossEntropy();
        Matrix g2 = ce.gradient(new Matrix(new double[][]{{0.5, 0.5}}), new Matrix(new double[][]{{1, 0}}));
        check("cross entropy gradient by hand", g2.approxEquals(new Matrix(new double[][]{{-2, 0}}), TOL));

        Matrix g3 = ce.gradient(new Matrix(new double[][]{{0.5, 0.5}, {0.25, 0.75}}), new Matrix(new double[][]{{1, 0}, {0, 1}}));
        check("cross entropy gradient batch factor", near(g3.get(0, 0), -1.0, TOL) && near(g3.get(1, 1), -1.0 / 0.75 / 2, TOL));

        Random rng = new Random(5);
        Matrix pred = Matrix.randomUniform(4, 3, 0.1, 0.9, rng);
        Matrix target = new Matrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 0}});
        checkAgainstNumerical("mse", mse, pred, target);
        checkAgainstNumerical("cross entropy", ce, pred, target);

        check("mse gradient keeps shape", mse.gradient(pred, target).shape().equals("4x3"));
    }

    private static void checkAgainstNumerical(String name, Loss loss, Matrix pred, Matrix target){
        Matrix analytical = loss.gradient(pred, target);
        double h = 1e-6;
        double worst = 0;
        for(int i = 0; i < pred.rows(); i++){
            for(int j = 0; j < pred.cols(); j++){
                double original = pred.get(i, j);
                pred.set(i, j, original + h);
                double plus = loss.compute(pred, target);
                pred.set(i, j, original - h);
                double minus = loss.compute(pred, target);
                pred.set(i, j, original);
                worst = Math.max(worst, Math.abs((plus - minus) / (2 * h) - analytical.get(i, j)));
            }
        }
        check(name + " gradient matches numerical slope", worst < 1e-6);
    }

    private static void testErrors(){
        Matrix a = Matrix.zeros(2, 3);
        Matrix b = Matrix.zeros(2, 2);
        expectThrows("mse shape mismatch", () -> new MeanSquaredError().compute(a, b));
        expectThrows("cross entropy shape mismatch", () -> new CrossEntropy().compute(a, b));
        expectThrows("mse gradient shape mismatch", () -> new MeanSquaredError().gradient(a, b));
        expectThrows("cross entropy gradient shape mismatch", () -> new CrossEntropy().gradient(a, b));
    }
}
