package neuralforge;

import neuralforge.math.Matrix;
import java.util.Random;

public class Main{

    public static void main(String[] args){
        System.out.println("NeuralForge - Matrix demo");

        Matrix x = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix w = new Matrix(new double[][]{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}});
        Matrix b = new Matrix(new double[][]{{0.5, -0.5}});

        Matrix z = x.multiply(w).addBias(b);

        System.out.println("X (" + x.shape() + "):");
        System.out.print(x);
        System.out.println("Z = X*W + b (" + z.shape() + "):");
        System.out.print(z);

        System.out.println("expected: [2.7 2.3] and [5.4 5.9]");

        Random rng = new Random(42);
        int n = 300;
        Matrix a = Matrix.randomUniform(n, n, -1, 1, rng);
        Matrix c = Matrix.randomUniform(n, n, -1, 1, rng);

        long start = System.nanoTime();
        a.multiply(c);
        long end = System.nanoTime();

        System.out.println("multiply " + n + "x" + n + " took " + (end - start) / 1_000_000 + " ms");
    }
}
