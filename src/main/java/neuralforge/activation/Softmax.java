package neuralforge.activation;

import neuralforge.math.Matrix;

public class Softmax implements Activation{

    @Override
    public Matrix forward(Matrix z){
        int rows = z.rows();
        int cols = z.cols();
        Matrix result = new Matrix(rows, cols);

        for(int i = 0; i < rows; i++){
            double max = z.get(i, 0);
            for(int j = 1; j < cols; j++){
                if(z.get(i, j) > max){
                    max = z.get(i, j);
                }
            }

            double sum = 0.0;
            for(int j = 0; j < cols; j++){
                double e = Math.exp(z.get(i, j) - max);
                result.set(i, j, e);
                sum += e;
            }

            for(int j = 0; j < cols; j++){
                result.set(i, j, result.get(i, j) / sum);
            }
        }
        return result;
    }

    @Override
    public Matrix backward(Matrix z, Matrix output, Matrix gradOutput){
        if(output.rows() != gradOutput.rows() || output.cols() != gradOutput.cols()){
            throw new IllegalArgumentException("gradient " + gradOutput.shape() + " does not match output " + output.shape());
        }
        int rows = output.rows();
        int cols = output.cols();
        Matrix result = new Matrix(rows, cols);

        for(int i = 0; i < rows; i++){
            double dot = 0.0;
            for(int j = 0; j < cols; j++){
                dot += gradOutput.get(i, j) * output.get(i, j);
            }
            for(int k = 0; k < cols; k++){
                result.set(i, k, output.get(i, k) * (gradOutput.get(i, k) - dot));
            }
        }
        return result;
    }

    @Override
    public String name(){
        return "softmax";
    }
}
