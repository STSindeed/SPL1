package neuralforge.preprocess;

import neuralforge.math.Matrix;
import java.util.Arrays;

public class Imputer{

    public enum Strategy{
        MEAN,
        MEDIAN
    }

    private final Strategy strategy;
    private double[] fill;

    public Imputer(Strategy strategy){
        if(strategy == null){
            throw new IllegalArgumentException("choose an imputation strategy");
        }
        this.strategy = strategy;
    }

    public void fit(Matrix data){
        int cols = data.cols();
        fill = new double[cols];

        for(int j = 0; j < cols; j++){
            double[] known = knownValues(data, j);
            if(known.length == 0){
                throw new IllegalArgumentException("column " + j + " has no known values, so there is nothing to fill its holes with");
            }
            fill[j] = strategy == Strategy.MEAN ? mean(known) : median(known);
        }
    }

    public Matrix transform(Matrix data){
        if(fill == null){
            throw new IllegalStateException("call fit() first");
        }
        if(data.cols() != fill.length){
            throw new IllegalArgumentException("imputer was fitted on " + fill.length + " columns but the data has " + data.cols());
        }
        Matrix result = new Matrix(data.rows(), data.cols());
        for(int i = 0; i < data.rows(); i++){
            for(int j = 0; j < data.cols(); j++){
                double value = data.get(i, j);
                result.set(i, j, Double.isNaN(value) ? fill[j] : value);
            }
        }
        return result;
    }

    public Matrix fitTransform(Matrix data){
        fit(data);
        return transform(data);
    }

    public double[] getFill(){
        if(fill == null){
            throw new IllegalStateException("call fit() first");
        }
        return fill.clone();
    }

    public Strategy getStrategy(){
        return strategy;
    }

    public static int countMissing(Matrix data, int column){
        int count = 0;
        for(int i = 0; i < data.rows(); i++){
            if(Double.isNaN(data.get(i, column))){
                count++;
            }
        }
        return count;
    }

    private static double[] knownValues(Matrix data, int column){
        double[] values = new double[data.rows()];
        int count = 0;
        for(int i = 0; i < data.rows(); i++){
            double value = data.get(i, column);
            if(!Double.isNaN(value)){
                values[count++] = value;
            }
        }
        return Arrays.copyOf(values, count);
    }

    private static double mean(double[] values){
        double sum = 0.0;
        for(double value : values){
            sum += value;
        }
        return sum / values.length;
    }

    private static double median(double[] values){
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int middle = sorted.length / 2;
        if(sorted.length % 2 == 1){
            return sorted[middle];
        }
        return (sorted[middle - 1] + sorted[middle]) / 2.0;
    }
}
