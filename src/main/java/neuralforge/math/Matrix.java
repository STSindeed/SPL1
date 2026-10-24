package neuralforge.math;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

public class Matrix{

    private final int rows;
    private final int cols;

    private final double[] data;

    public Matrix(int rows, int cols){
        // a matrix with zero or negative size makes no sense, so fail early
        if(rows <= 0 || cols <= 0){
            throw new IllegalArgumentException("size must be positive, got " + rows + "x" + cols);
        }
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows * cols];
    }

    public Matrix(double[][] values){
        this(values.length, values[0].length);
        for(int i = 0; i < rows; i++){
            if(values[i].length != cols){
                throw new IllegalArgumentException("row " + i + " has a different length");
            }
            System.arraycopy(values[i], 0, data, i * cols, cols);
        }
    }

    public static Matrix zeros(int rows, int cols){
        return new Matrix(rows, cols);
    }

    public static Matrix filled(int rows, int cols, double value){
        Matrix m = new Matrix(rows, cols);
        Arrays.fill(m.data, value);
        return m;
    }

    public static Matrix identity(int n){
        Matrix m = new Matrix(n, n);
        for(int i = 0; i < n; i++){
            m.data[i * n + i] = 1.0;
        }
        return m;
    }

    public static Matrix randomUniform(int rows, int cols, double low, double high, Random rng){
        Matrix m = new Matrix(rows, cols);
        for(int i = 0; i < m.data.length; i++){
            m.data[i] = low + (high - low) * rng.nextDouble();
        }
        return m;
    }

    public static Matrix randomGaussian(int rows, int cols, double std, Random rng){
        Matrix m = new Matrix(rows, cols);
        for(int i = 0; i < m.data.length; i++){
            m.data[i] = rng.nextGaussian() * std;
        }
        return m;
    }

    public int rows(){
        return rows;
    }

    public int cols(){
        return cols;
    }

    public double get(int i, int j){
        checkIndex(i, j);
        return data[i * cols + j];
    }

    public void set(int i, int j, double value){
        checkIndex(i, j);
        data[i * cols + j] = value;
    }

    public Matrix copy(){
        Matrix m = new Matrix(rows, cols);
        System.arraycopy(data, 0, m.data, 0, data.length);
        return m;
    }

    public Matrix multiply(Matrix other){
        if(this.cols != other.rows){
            throw new IllegalArgumentException("cannot multiply " + shape() + " with " + other.shape());
        }

        int n = this.rows;
        int m = this.cols;
        int p = other.cols;

        Matrix result = new Matrix(n, p);

        for(int i = 0; i < n; i++){
            for(int k = 0; k < m; k++){
                double a = this.data[i * m + k];
                for(int j = 0; j < p; j++){
                    result.data[i * p + j] += a * other.data[k * p + j];
                }
            }
        }
        return result;
    }

    public Matrix transpose(){
        Matrix result = new Matrix(cols, rows);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                result.data[j * rows + i] = this.data[i * cols + j];
            }
        }
        return result;
    }

    public Matrix add(Matrix other){
        checkSameShape(other, "add");
        Matrix result = new Matrix(rows, cols);
        for(int i = 0; i < data.length; i++){
            result.data[i] = this.data[i] + other.data[i];
        }
        return result;
    }

    public Matrix subtract(Matrix other){
        checkSameShape(other, "subtract");
        Matrix result = new Matrix(rows, cols);
        for(int i = 0; i < data.length; i++){
            result.data[i] = this.data[i] - other.data[i];
        }
        return result;
    }

    public Matrix hadamard(Matrix other){
        checkSameShape(other, "hadamard");
        Matrix result = new Matrix(rows, cols);
        for(int i = 0; i < data.length; i++){
            result.data[i] = this.data[i] * other.data[i];
        }
        return result;
    }

    public Matrix scale(double factor){
        Matrix result = new Matrix(rows, cols);
        for(int i = 0; i < data.length; i++){
            result.data[i] = this.data[i] * factor;
        }
        return result;
    }

    public Matrix apply(DoubleUnaryOperator f){
        Matrix result = new Matrix(rows, cols);
        for(int i = 0; i < data.length; i++){
            result.data[i] = f.applyAsDouble(this.data[i]);
        }
        return result;
    }

    public void addScaledInPlace(Matrix other, double factor){
        checkSameShape(other, "addScaledInPlace");
        for(int i = 0; i < data.length; i++){
            this.data[i] += factor * other.data[i];
        }
    }

    public Matrix addBias(Matrix bias){
        if(bias.rows != 1 || bias.cols != this.cols){
            throw new IllegalArgumentException("bias must be 1x" + cols + " but it is " + bias.shape());
        }
        Matrix result = new Matrix(rows, cols);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                result.data[i * cols + j] = this.data[i * cols + j] + bias.data[j];
            }
        }
        return result;
    }

    public Matrix columnSums(){
        Matrix result = new Matrix(1, cols);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                result.data[j] += this.data[i * cols + j];
            }
        }
        return result;
    }

    public Matrix selectRows(int[] indices, int from, int to){
        if(from < 0 || to > indices.length || from >= to){
            throw new IllegalArgumentException("bad range " + from + ".." + to + " for " + indices.length + " indices");
        }
        Matrix result = new Matrix(to - from, cols);
        for(int r = from; r < to; r++){
            int source = indices[r];
            if(source < 0 || source >= rows){
                throw new IndexOutOfBoundsException("row " + source + " is outside " + shape());
            }
            System.arraycopy(this.data, source * cols, result.data, (r - from) * cols, cols);
        }
        return result;
    }

    public int[] argmaxRows(){
        int[] best = new int[rows];
        for(int i = 0; i < rows; i++){
            int bestCol = 0;
            double bestValue = data[i * cols];
            for(int j = 1; j < cols; j++){
                if(data[i * cols + j] > bestValue){
                    bestValue = data[i * cols + j];
                    bestCol = j;
                }
            }
            best[i] = bestCol;
        }
        return best;
    }

    public boolean approxEquals(Matrix other, double tolerance){
        if(this.rows != other.rows || this.cols != other.cols){
            return false;
        }
        for(int i = 0; i < data.length; i++){
            if(Math.abs(this.data[i] - other.data[i]) > tolerance){
                return false;
            }
        }
        return true;
    }

    public String shape(){
        return rows + "x" + cols;
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < rows; i++){
            sb.append("[ ");
            for(int j = 0; j < cols; j++){
                sb.append(String.format(Locale.US, "%9.4f ", data[i * cols + j]));
            }
            sb.append("]\n");
        }
        return sb.toString();
    }

    private void checkIndex(int i, int j){
        if(i < 0 || i >= rows || j < 0 || j >= cols){
            throw new IndexOutOfBoundsException("(" + i + ", " + j + ") is outside " + shape());
        }
    }

    private void checkSameShape(Matrix other, String opName){
        if(this.rows != other.rows || this.cols != other.cols){
            throw new IllegalArgumentException("cannot " + opName + " " + shape() + " and " + other.shape());
        }
    }
}
