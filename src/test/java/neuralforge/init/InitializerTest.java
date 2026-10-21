package neuralforge.init;

import neuralforge.math.Matrix;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class InitializerTest{

    public static void main(String[] args){
        testXavier();
        testHe();
        testReproducible();
        finish();
    }

    private static double mean(Matrix m){
        double sum = 0;
        for(int i = 0; i < m.rows(); i++){
            for(int j = 0; j < m.cols(); j++){
                sum += m.get(i, j);
            }
        }
        return sum / (m.rows() * m.cols());
    }

    private static double std(Matrix m){
        double mu = mean(m);
        double sum = 0;
        for(int i = 0; i < m.rows(); i++){
            for(int j = 0; j < m.cols(); j++){
                double d = m.get(i, j) - mu;
                sum += d * d;
            }
        }
        return Math.sqrt(sum / (m.rows() * m.cols()));
    }

    private static double maxAbs(Matrix m){
        double best = 0;
        for(int i = 0; i < m.rows(); i++){
            for(int j = 0; j < m.cols(); j++){
                best = Math.max(best, Math.abs(m.get(i, j)));
            }
        }
        return best;
    }

    private static void testXavier(){
        int fanIn = 100;
        int fanOut = 300;
        Matrix w = new XavierInitializer().createWeights(fanIn, fanOut, new Random(3));

        check("xavier shape", w.rows() == fanIn && w.cols() == fanOut);

        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        check("xavier stays inside the limit", maxAbs(w) <= limit);

        check("xavier uses the whole range", maxAbs(w) > 0.95 * limit);

        check("xavier mean near 0", near(mean(w), 0.0, 0.005));
        double expectedStd = limit / Math.sqrt(3.0);
        check("xavier std", near(std(w), expectedStd, 0.05 * expectedStd));
    }

    private static void testHe(){
        int fanIn = 200;
        int fanOut = 200;
        Matrix w = new HeInitializer().createWeights(fanIn, fanOut, new Random(4));

        check("he shape", w.rows() == fanIn && w.cols() == fanOut);

        double expectedStd = Math.sqrt(2.0 / fanIn);
        check("he mean near 0", near(mean(w), 0.0, 0.005));
        check("he std", near(std(w), expectedStd, 0.05 * expectedStd));

        Matrix wide = new HeInitializer().createWeights(fanIn, 400, new Random(4));
        check("he ignores fanOut", near(std(wide), expectedStd, 0.05 * expectedStd));
    }

    private static void testReproducible(){
        Matrix a = new HeInitializer().createWeights(10, 10, new Random(9));
        Matrix b = new HeInitializer().createWeights(10, 10, new Random(9));
        check("same seed, same weights", a.approxEquals(b, 0.0));

        Matrix c = new HeInitializer().createWeights(10, 10, new Random(10));
        check("different seed, different weights", !a.approxEquals(c, 1e-12));
    }
}
