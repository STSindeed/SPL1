package neuralforge.math;

import java.util.Random;
import static neuralforge.TestHelper.*;

/*
 * Tests for the Matrix class.
 * It uses the small TestHelper class instead of JUnit, so the project needs
 * no external libraries. Run it with: ./build.sh test
 */
public class MatrixTest{

    public static void main(String[] args){
        // each method below tests one group of features
        testConstructors();
        testMultiply();
        testTranspose();
        testElementWise();
        testBias();
        testAgainstNaive();

        // prints the totals and sets the exit code
        finish();
    }

    // ---------- tiny helpers ----------

    // Short name for creating a matrix from a 2D array, keeps tests compact.
    private static Matrix m(double[][] values){
        return new Matrix(values);
    }

    // ---------- tests ----------

    private static void testConstructors(){
        Matrix z = Matrix.zeros(2, 3);
        check("zeros shape", z.rows() == 2 && z.cols() == 3);
        check("zeros value", z.get(1, 2) == 0.0);

        // identity: ones on the diagonal, zeros elsewhere
        Matrix id = Matrix.identity(3);
        check("identity diagonal", id.get(0, 0) == 1.0 && id.get(2, 2) == 1.0);
        check("identity off diagonal", id.get(0, 1) == 0.0);

        Matrix f = Matrix.filled(2, 2, 7.5);
        check("filled", f.get(1, 1) == 7.5);

        // changing the copy must not touch the original
        Matrix copy = id.copy();
        copy.set(0, 0, 9.0);
        check("copy is independent", id.get(0, 0) == 1.0);

        // invalid input must be rejected
        expectThrows("zero rows", () -> new Matrix(0, 3));
        expectThrows("ragged array", () -> m(new double[][]{{1, 2}, {3}}));
        expectThrows("get outside", () -> z.get(2, 0));
        expectThrows("get column outside", () -> z.get(0, 3));
    }

    private static void testMultiply(){
        Matrix a = m(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix b = m(new double[][]{{7, 8}, {9, 10}, {11, 12}});

        // worked out by hand, e.g. top-left = 1*7 + 2*9 + 3*11 = 58
        Matrix expected = m(new double[][]{{58, 64}, {139, 154}});
        check("multiply 2x3 * 3x2", a.multiply(b).approxEquals(expected, 1e-9));

        // identity must leave the matrix unchanged
        check("multiply by identity", a.multiply(Matrix.identity(3)).approxEquals(a, 1e-9));
        check("multiply result shape", a.multiply(b).shape().equals("2x2"));

        // matrix multiplication is not commutative (here the shapes even differ)
        check("b*a differs from a*b", !b.multiply(a).approxEquals(a.multiply(b), 1e-9));

        // 2x3 times 2x3 is not allowed
        expectThrows("multiply bad shapes", () -> a.multiply(a));
    }

    private static void testTranspose(){
        Matrix a = m(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix t = a.transpose();
        check("transpose shape", t.rows() == 3 && t.cols() == 2);
        check("transpose values", t.get(2, 0) == 3.0 && t.get(0, 1) == 4.0);

        // transposing twice gives back the original
        check("transpose twice", t.transpose().approxEquals(a, 1e-12));

        // known identity: (AB)^T = B^T A^T
        Matrix b = m(new double[][]{{7, 8}, {9, 10}, {11, 12}});
        Matrix left = a.multiply(b).transpose();
        Matrix right = b.transpose().multiply(a.transpose());
        check("(AB)^T == B^T A^T", left.approxEquals(right, 1e-9));
    }

    private static void testElementWise(){
        Matrix a = m(new double[][]{{1, 2}, {3, 4}});
        Matrix b = m(new double[][]{{5, 6}, {7, 8}});

        check("add", a.add(b).approxEquals(m(new double[][]{{6, 8}, {10, 12}}), 1e-9));
        check("subtract", b.subtract(a).approxEquals(m(new double[][]{{4, 4}, {4, 4}}), 1e-9));
        check("hadamard", a.hadamard(b).approxEquals(m(new double[][]{{5, 12}, {21, 32}}), 1e-9));
        check("scale", a.scale(2).approxEquals(m(new double[][]{{2, 4}, {6, 8}}), 1e-9));
        check("apply square", a.apply(v -> v * v).approxEquals(m(new double[][]{{1, 4}, {9, 16}}), 1e-9));

        // none of the operations above may modify a
        check("original untouched", a.get(0, 0) == 1.0);

        // different shapes must be rejected
        expectThrows("add bad shapes", () -> a.add(Matrix.zeros(3, 2)));
        expectThrows("hadamard bad shapes", () -> a.hadamard(Matrix.zeros(2, 3)));
    }

    private static void testBias(){
        Matrix z = m(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix bias = m(new double[][]{{10, 20, 30}});

        // the same bias row is added to both rows
        Matrix expected = m(new double[][]{{11, 22, 33}, {14, 25, 36}});
        check("addBias", z.addBias(bias).approxEquals(expected, 1e-9));

        // column totals: 1+4, 2+5, 3+6
        check("columnSums", z.columnSums().approxEquals(m(new double[][]{{5, 7, 9}}), 1e-9));

        expectThrows("bias wrong width", () -> z.addBias(Matrix.zeros(1, 2)));
        expectThrows("bias not a row", () -> z.addBias(Matrix.zeros(2, 3)));
    }

    // Compares the fast i-k-j multiply with the plain i-j-k version on many
    // random matrices of random sizes. If both agree, the fast one is correct.
    private static void testAgainstNaive(){
        Random rng = new Random(7);
        for(int trial = 0; trial < 20; trial++){
            // sizes between 1 and 12
            int n = 1 + rng.nextInt(12);
            int k = 1 + rng.nextInt(12);
            int p = 1 + rng.nextInt(12);
            Matrix a = Matrix.randomUniform(n, k, -5, 5, rng);
            Matrix b = Matrix.randomUniform(k, p, -5, 5, rng);
            check("random multiply " + n + "x" + k + "x" + p, a.multiply(b).approxEquals(naiveMultiply(a, b), 1e-9));
        }
    }

    // The slow textbook version, used only as a reference answer.
    // Each result element is the dot product of a row of a and a column of b.
    private static Matrix naiveMultiply(Matrix a, Matrix b){
        Matrix r = new Matrix(a.rows(), b.cols());
        for(int i = 0; i < a.rows(); i++){
            for(int j = 0; j < b.cols(); j++){
                double sum = 0;
                for(int k = 0; k < a.cols(); k++){
                    sum += a.get(i, k) * b.get(k, j);
                }
                r.set(i, j, sum);
            }
        }
        return r;
    }
}
