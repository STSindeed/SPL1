package neuralforge.gui;

public class Prediction{

    private final double[] probabilities;
    private final int best;

    public Prediction(double[] probabilities){
        if(probabilities.length == 0){
            throw new IllegalArgumentException("a prediction needs at least one class");
        }
        this.probabilities = probabilities.clone();

        int bestIndex = 0;
        for(int i = 1; i < probabilities.length; i++){
            if(probabilities[i] > probabilities[bestIndex]){
                bestIndex = i;
            }
        }
        this.best = bestIndex;
    }

    public int digit(){
        return best;
    }

    public double confidence(){
        return probabilities[best];
    }

    public double probability(int digit){
        return probabilities[digit];
    }

    public double[] getProbabilities(){
        return probabilities.clone();
    }

    public int[] top(int k){
        if(k < 1 || k > probabilities.length){
            throw new IllegalArgumentException("k must be between 1 and " + probabilities.length + ", got " + k);
        }
        boolean[] used = new boolean[probabilities.length];
        int[] result = new int[k];
        for(int rank = 0; rank < k; rank++){
            int pick = -1;
            for(int i = 0; i < probabilities.length; i++){
                if(!used[i] && (pick == -1 || probabilities[i] > probabilities[pick])){
                    pick = i;
                }
            }
            used[pick] = true;
            result[rank] = pick;
        }
        return result;
    }
}
