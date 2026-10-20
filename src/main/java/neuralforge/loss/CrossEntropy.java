package neuralforge.loss;

import neuralforge.math.Matrix;

public class CrossEntropy implements Loss{


    private static final double EPSILON = 1e-12;

    @Override
    public double compute(Matrix predicted, Matrix target){
        if(predicted.rows() != target.rows() || predicted.cols() != target.cols()){
            throw new IllegalArgumentException("shape mismatch: predicted " + predicted.shape() + ", target " + target.shape());
        }

        double sum = 0.0;
        for(int i = 0; i < predicted.rows(); i++){
            for(int j = 0; j < predicted.cols(); j++){
                double t = target.get(i, j);

                if(t == 0.0){
                    continue;
                }

                double p = Math.max(predicted.get(i, j), EPSILON);
                sum += t * Math.log(p);
            }
        }

        return -sum / predicted.rows();
    }

    @Override
    public String name(){
        return "cross_entropy";
    }
}
