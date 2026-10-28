package neuralforge.preprocess;

import neuralforge.math.Matrix;

public class StandardScaler implements Scaler{

    private static final double MIN_STD = 1e-12;

    private double[] mean;
    private double[] std;

    @Override
    public void fit(Matrix data){
        int rows = data.rows();
        int cols = data.cols();
        mean = new double[cols];
        std = new double[cols];

        for(int j = 0; j < cols; j++){
            double sum = 0.0;
            for(int i = 0; i < rows; i++){
                sum += data.get(i, j);
            }
            mean[j] = sum / rows;

            double squares = 0.0;
            for(int i = 0; i < rows; i++){
                double diff = data.get(i, j) - mean[j];
                squares += diff * diff;
            }
            std[j] = Math.sqrt(squares / rows);

            if(std[j] < MIN_STD){
                std[j] = 1.0;
            }
        }
    }

    @Override
    public Matrix transform(Matrix data){
        checkReady(data);
        Matrix result = new Matrix(data.rows(), data.cols());
        for(int i = 0; i < data.rows(); i++){
            for(int j = 0; j < data.cols(); j++){
                result.set(i, j, (data.get(i, j) - mean[j]) / std[j]);
            }
        }
        return result;
    }

    public Matrix inverseTransform(Matrix data){
        checkReady(data);
        Matrix result = new Matrix(data.rows(), data.cols());
        for(int i = 0; i < data.rows(); i++){
            for(int j = 0; j < data.cols(); j++){
                result.set(i, j, data.get(i, j) * std[j] + mean[j]);
            }
        }
        return result;
    }

    public double[] getMean(){
        checkFitted();
        return mean.clone();
    }

    public double[] getStd(){
        checkFitted();
        return std.clone();
    }

    @Override
    public String name(){
        return "standard";
    }

    private void checkFitted(){
        if(mean == null){
            throw new IllegalStateException("call fit() first");
        }
    }

    private void checkReady(Matrix data){
        checkFitted();
        if(data.cols() != mean.length){
            throw new IllegalArgumentException("scaler was fitted on " + mean.length + " columns but the data has " + data.cols());
        }
    }
}
