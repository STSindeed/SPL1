package neuralforge.data;

import neuralforge.math.Matrix;
import java.util.Locale;
import java.util.Random;

public class Dataset{

    private final Matrix features;
    private final int[] labels;
    private final String[] classNames;

    private Matrix targets;

    public Dataset(Matrix features, int[] labels, String[] classNames){
        if(features.rows() != labels.length){
            throw new IllegalArgumentException("there are " + features.rows() + " feature rows but " + labels.length + " labels");
        }
        if(classNames.length == 0){
            throw new IllegalArgumentException("a dataset needs at least one class");
        }
        for(int i = 0; i < labels.length; i++){
            if(labels[i] < 0 || labels[i] >= classNames.length){
                throw new IllegalArgumentException("label " + labels[i] + " of sample " + i + " is outside 0.." + (classNames.length - 1));
            }
        }
        this.features = features;
        this.labels = labels.clone();
        this.classNames = classNames.clone();
    }

    public int size(){
        return features.rows();
    }

    public int featureCount(){
        return features.cols();
    }

    public int classCount(){
        return classNames.length;
    }

    public Matrix getFeatures(){
        return features;
    }

    public int[] getLabels(){
        return labels.clone();
    }

    public String[] getClassNames(){
        return classNames.clone();
    }

    public String className(int index){
        return classNames[index];
    }

    public Matrix getTargets(){
        if(targets == null){
            Matrix result = new Matrix(size(), classCount());
            for(int i = 0; i < labels.length; i++){
                result.set(i, labels[i], 1.0);
            }
            targets = result;
        }
        return targets;
    }

    public int[] classCounts(){
        int[] counts = new int[classCount()];
        for(int label : labels){
            counts[label]++;
        }
        return counts;
    }

    public Dataset subset(int[] indices){
        Matrix pickedFeatures = features.selectRows(indices, 0, indices.length);
        int[] pickedLabels = new int[indices.length];
        for(int i = 0; i < indices.length; i++){
            pickedLabels[i] = labels[indices[i]];
        }
        return new Dataset(pickedFeatures, pickedLabels, classNames);
    }

    public Dataset withFeatures(Matrix newFeatures){
        if(newFeatures.rows() != features.rows()){
            throw new IllegalArgumentException("new features have " + newFeatures.rows() + " rows but the dataset has " + features.rows());
        }
        return new Dataset(newFeatures, labels, classNames);
    }

    public Dataset[] stratifiedSplit(double testFraction, Random rng){
        if(!(testFraction > 0.0 && testFraction < 1.0)){
            throw new IllegalArgumentException("testFraction must be between 0 and 1, got " + testFraction);
        }

        int classes = classCount();
        int[] counts = classCounts();

        int[][] byClass = new int[classes][];
        for(int c = 0; c < classes; c++){
            byClass[c] = new int[counts[c]];
        }
        int[] filled = new int[classes];
        for(int i = 0; i < labels.length; i++){
            int c = labels[i];
            byClass[c][filled[c]] = i;
            filled[c]++;
        }

        int[] testCounts = new int[classes];
        int totalTest = 0;
        for(int c = 0; c < classes; c++){
            if(counts[c] >= 2){
                int wanted = (int) Math.round(counts[c] * testFraction);
                testCounts[c] = Math.max(1, Math.min(wanted, counts[c] - 1));
            }
            totalTest += testCounts[c];
        }

        if(totalTest == 0){
            throw new IllegalArgumentException("not enough samples to make a test set");
        }

        int[] testRows = new int[totalTest];
        int[] trainRows = new int[size() - totalTest];
        int testPos = 0;
        int trainPos = 0;
        for(int c = 0; c < classes; c++){
            Shuffler.shuffle(byClass[c], rng);
            for(int k = 0; k < byClass[c].length; k++){
                if(k < testCounts[c]){
                    testRows[testPos++] = byClass[c][k];
                }else{
                    trainRows[trainPos++] = byClass[c][k];
                }
            }
        }

        Shuffler.shuffle(trainRows, rng);
        Shuffler.shuffle(testRows, rng);

        return new Dataset[]{subset(trainRows), subset(testRows)};
    }

    public String describe(){
        StringBuilder sb = new StringBuilder();
        sb.append(size()).append(" samples, ").append(featureCount()).append(" features, ");
        sb.append(classCount()).append(" classes\n");
        int[] counts = classCounts();
        for(int c = 0; c < classCount(); c++){
            double percent = 100.0 * counts[c] / size();
            sb.append(String.format(Locale.US, "  %-12s %6d  (%.1f%%)\n", classNames[c], counts[c], percent));
        }
        return sb.toString();
    }
}
