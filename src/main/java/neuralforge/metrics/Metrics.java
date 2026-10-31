package neuralforge.metrics;

import neuralforge.math.Matrix;

public class Metrics{

    private Metrics(){
    }

    public static double topKAccuracy(Matrix scores, int[] labels, int k){
        if(scores.rows() != labels.length){
            throw new IllegalArgumentException("there are " + scores.rows() + " score rows but " + labels.length + " labels");
        }
        if(k < 1 || k > scores.cols()){
            throw new IllegalArgumentException("k must be between 1 and " + scores.cols() + ", got " + k);
        }

        int hits = 0;
        for(int i = 0; i < labels.length; i++){
            int label = labels[i];
            if(label < 0 || label >= scores.cols()){
                throw new IllegalArgumentException("label " + label + " of sample " + i + " is outside 0.." + (scores.cols() - 1));
            }

            double trueScore = scores.get(i, label);
            int better = 0;
            for(int j = 0; j < scores.cols(); j++){
                if(scores.get(i, j) > trueScore){
                    better++;
                }
            }
            if(better < k){
                hits++;
            }
        }
        return (double) hits / labels.length;
    }
}
