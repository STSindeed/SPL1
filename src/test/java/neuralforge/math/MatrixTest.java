package neuralforge.math;

import java.util.Random;

public class MatrixTest{

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args){
        testConstructors();
        testMultiply();
        testTranspose();
        testElementWise();
        testBias();
        testAgainstNaive();

        System.out.println();
        System.out.println(passed + " passed, " + failed + " failed");

        if(failed > 0){
            System.exit(1);
        }
    }

    private static void check(String name, boolean condition){
        if(condition){
            passed++;
        }else{
            failed++;
            System.out.println("FAIL: " + name);
        }
    }

    private static void expectThrows(String name, Runnable code){
        try{
            code.run();
            failed++;
            System.out.println("FAIL (no exception): " + name);
        }catch(RuntimeException e){
            passed++;
        }
    }

    private static Matrix m(double[][] values){
        return new Matrix(values);
    }

    private static void testConstructors(){
        Matrix z = Matrix.zeros(2, 3);
        check("zeros shape", z.rows() == 2 && z.cols() == 3);
        check("zeros value", z.get(1, 2) == 0.0);

        Matrix id = Matrix.identity(3);
        check("identity diagonal", id.get(0, 0) == 1.0 && id.get(2, 2) == 1.0);
        check("identity off diagonal", id.get(0, 1) == 0.0);

        Matrix f = Matrix.filled(2, 2, 7.5);
        check("filled", f.get(1, 1) == 7.5);

        Matrix copy = id.copy();
        copy.set(0, 0, 9.0);
        check("copy is independent", id.get(0, 0) == 1.0);

        expectThrows("zero rows", () -> new Matrix(0, 3));
        expectThrows("ragged array", () -> m(new double[][]{{1, 2}, {3}}));
        expectThrows("get outside", () -> z.get(2, 0));
        expectThrows("get column outside", () -> z.get(0, 3));
    }

    private static void testMultiply(){
        Matrix a = m(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix b = m(new double[][]{{7, 8}, {9, 10}, {11, 12}});

        Matrix expected = m(new double[][]{{58, 64}, {139, 154}});
        check("multiply 2x3 * 3x2", a.multiply(b).approxEquals(expected, 1e-9));

        check("multiply by identity", a.multiply(Matrix.identity(3)).approxEquals(a, 1e-9));
        check("multiply result shape", a.multiply(b).shape().equals("2x2"));

        check("b*a differs from a*b", !b.multiply(a).approxEquals(a.multiply(b), 1e-9));

        expectThrows("multiply bad shapes", () -> a.multiply(a));
    }

    private static void testTranspose(){
        Matrix a = m(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix t = a.transpose();
        check("transpose shape", t.rows() == 3 && t.cols() == 2);
        check("transpose values", t.get(2, 0) == 3.0 && t.get(0, 1) == 4.0);

        check("transpose twice", t.transpose().approxEquals(a, 1e-12));

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

        check("original untouched", a.get(0, 0) == 1.0);

        expectThrows("add bad shapes", () -> a.add(Matrix.zeros(3, 2)));
        expectThrows("hadamard bad shapes", () -> a.hadamard(Matrix.zeros(2, 3)));
    }

    private static void testBias(){
        Matrix z = m(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix bias = m(new double[][]{{10, 20, 30}});

        Matrix expected = m(new double[][]{{11, 22, 33}, {14, 25, 36}});
        check("addBias", z.addBias(bias).approxEquals(expected, 1e-9));

        check("columnSums", z.columnSums().approxEquals(m(new double[][]{{5, 7, 9}}), 1e-9));

        expectThrows("bias wrong width", () -> z.addBias(Matrix.zeros(1, 2)));
        expectThrows("bias not a row", () -> z.addBias(Matrix.zeros(2, 3)));
    }

    private static void testAgainstNaive(){
        Random rng = new Random(7);
        for(int trial = 0; trial < 20; trial++){
            int n = 1 + rng.nextInt(12);
            int k = 1 + rng.nextInt(12);
            int p = 1 + rng.nextInt(12);
            Matrix a = Matrix.randomUniform(n, k, -5, 5, rng);
            Matrix b = Matrix.randomUniform(k, p, -5, 5, rng);
            check("random multiply " + n + "x" + k + "x" + p, a.multiply(b).approxEquals(naiveMultiply(a, b), 1e-9));
        }
    }

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
