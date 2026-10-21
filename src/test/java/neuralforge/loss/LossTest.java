package neuralforge.loss;

import neuralforge.math.Matrix;
import static neuralforge.TestHelper.*;

public class LossTest{

    private static final double TOL = 1e-9;

    public static void main(String[] args){
        testMeanSquaredError();
        testCrossEntropy();
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

    private static void testErrors(){
        Matrix a = Matrix.zeros(2, 3);
        Matrix b = Matrix.zeros(2, 2);
        expectThrows("mse shape mismatch", () -> new MeanSquaredError().compute(a, b));
        expectThrows("cross entropy shape mismatch", () -> new CrossEntropy().compute(a, b));
    }
}
