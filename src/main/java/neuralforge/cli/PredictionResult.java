package neuralforge.cli;

import neuralforge.math.Matrix;

public class PredictionResult{

    private final int[] classes;
    private final Matrix probabilities;
    private final String[] classNames;

    public PredictionResult(int[] classes, Matrix probabilities, String[] classNames){
        if(classes.length != probabilities.rows() || probabilities.cols() != classNames.length){
            throw new IllegalArgumentException("the classes, probabilities and class names do not fit together");
        }
        this.classes = classes;
        this.probabilities = probabilities;
        this.classNames = classNames;
    }

    public int size(){
        return classes.length;
    }

    public int classOf(int row){
        return classes[row];
    }

    public String className(int row){
        return classNames[classes[row]];
    }

    public double confidence(int row){
        return probabilities.get(row, classes[row]);
    }

    public double probability(int row, int classIndex){
        return probabilities.get(row, classIndex);
    }

    public int[] getClasses(){
        return classes.clone();
    }

    public String[] getClassNames(){
        return classNames.clone();
    }

    public int[] classCounts(){
        int[] counts = new int[classNames.length];
        for(int c : classes){
            counts[c]++;
        }
        return counts;
    }

    public PredictionResult select(int[] positions){
        int[] chosen = new int[positions.length];
        Matrix chosenProbabilities = new Matrix(Math.max(1, positions.length), classNames.length);
        for(int i = 0; i < positions.length; i++){
            chosen[i] = classes[positions[i]];
            for(int c = 0; c < classNames.length; c++){
                chosenProbabilities.set(i, c, probabilities.get(positions[i], c));
            }
        }
        if(positions.length == 0){
            throw new IllegalArgumentException("nothing was selected");
        }
        return new PredictionResult(chosen, chosenProbabilities, classNames);
    }
}
