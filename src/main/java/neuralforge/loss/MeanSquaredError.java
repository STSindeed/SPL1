package neuralforge.loss;

import neuralforge.math.Matrix;

public class MeanSquaredError implements Loss{

    @Override
    public double compute(Matrix predicted, Matrix target){
        if(predicted.rows() != target.rows() || predicted.cols() != target.cols()){
            throw new IllegalArgumentException("shape mismatch: predicted " + predicted.shape() + ", target " + target.shape());
        }

        double sum = 0.0;
        for(int i = 0; i < predicted.rows(); i++){
            for(int j = 0; j < predicted.cols(); j++){
                double diff = predicted.get(i, j) - target.get(i, j);
                sum += diff * diff;
            }
        }

        return sum / (predicted.rows() * predicted.cols());
    }

    @Override
    public String name(){
        return "mse";
    }
}
