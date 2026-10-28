package neuralforge.preprocess;

import neuralforge.math.Matrix;

public class MinMaxScaler implements Scaler{

    private double[] min;
    private double[] range;

    @Override
    public void fit(Matrix data){
        int rows = data.rows();
        int cols = data.cols();
        min = new double[cols];
        range = new double[cols];

        for(int j = 0; j < cols; j++){
            double low = data.get(0, j);
            double high = low;
            for(int i = 1; i < rows; i++){
                double value = data.get(i, j);
                if(value < low){
                    low = value;
                }
                if(value > high){
                    high = value;
                }
            }
            min[j] = low;
            range[j] = high - low;

            if(range[j] == 0.0){
                range[j] = 1.0;
            }
        }
    }

    @Override
    public Matrix transform(Matrix data){
        if(min == null){
            throw new IllegalStateException("call fit() first");
        }
        if(data.cols() != min.length){
            throw new IllegalArgumentException("scaler was fitted on " + min.length + " columns but the data has " + data.cols());
        }
        Matrix result = new Matrix(data.rows(), data.cols());
        for(int i = 0; i < data.rows(); i++){
            for(int j = 0; j < data.cols(); j++){
                result.set(i, j, (data.get(i, j) - min[j]) / range[j]);
            }
        }
        return result;
    }

    @Override
    public String name(){
        return "minmax";
    }
}
