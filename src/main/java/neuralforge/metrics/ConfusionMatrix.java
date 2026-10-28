package neuralforge.metrics;

import java.util.Locale;

public class ConfusionMatrix{

    private final int classCount;

    private final int[][] counts;
    private int total = 0;

    public ConfusionMatrix(int classCount){
        if(classCount <= 0){
            throw new IllegalArgumentException("classCount must be positive, got " + classCount);
        }
        this.classCount = classCount;
        this.counts = new int[classCount][classCount];
    }

    public static ConfusionMatrix of(int[] actual, int[] predicted, int classCount){
        ConfusionMatrix matrix = new ConfusionMatrix(classCount);
        matrix.addAll(actual, predicted);
        return matrix;
    }

    public void add(int actual, int predicted){
        checkClass(actual);
        checkClass(predicted);
        counts[actual][predicted]++;
        total++;
    }

    public void addAll(int[] actual, int[] predicted){
        if(actual.length != predicted.length){
            throw new IllegalArgumentException("got " + actual.length + " actual and " + predicted.length + " predicted values");
        }
        for(int i = 0; i < actual.length; i++){
            add(actual[i], predicted[i]);
        }
    }

    public int get(int actual, int predicted){
        checkClass(actual);
        checkClass(predicted);
        return counts[actual][predicted];
    }

    public int total(){
        return total;
    }

    public int support(int c){
        checkClass(c);
        int sum = 0;
        for(int p = 0; p < classCount; p++){
            sum += counts[c][p];
        }
        return sum;
    }

    private int predictedAs(int c){
        int sum = 0;
        for(int a = 0; a < classCount; a++){
            sum += counts[a][c];
        }
        return sum;
    }

    public double accuracy(){
        if(total == 0){
            return 0.0;
        }
        int correct = 0;
        for(int c = 0; c < classCount; c++){
            correct += counts[c][c];
        }
        return (double) correct / total;
    }

    public double precision(int c){
        checkClass(c);
        int predicted = predictedAs(c);
        if(predicted == 0){
            return 0.0;
        }
        return (double) counts[c][c] / predicted;
    }

    public double recall(int c){
        checkClass(c);
        int actual = support(c);
        if(actual == 0){
            return 0.0;
        }
        return (double) counts[c][c] / actual;
    }

    public double f1(int c){
        double p = precision(c);
        double r = recall(c);
        if(p + r == 0.0){
            return 0.0;
        }
        return 2.0 * p * r / (p + r);
    }

    public double macroF1(){
        double sum = 0.0;
        int used = 0;
        for(int c = 0; c < classCount; c++){
            if(support(c) > 0){
                sum += f1(c);
                used++;
            }
        }
        if(used == 0){
            return 0.0;
        }
        return sum / used;
    }

    public String format(String[] classNames){
        checkNames(classNames);

        int labelWidth = 0;
        String[] labels = new String[classCount];
        for(int c = 0; c < classCount; c++){
            if(String.valueOf(c).equals(classNames[c])){
                labels[c] = classNames[c];
            }else{
                labels[c] = c + " " + classNames[c];
            }
            labelWidth = Math.max(labelWidth, labels[c].length());
        }

        int cellWidth = String.valueOf(classCount - 1).length();
        for(int a = 0; a < classCount; a++){
            for(int p = 0; p < classCount; p++){
                cellWidth = Math.max(cellWidth, String.valueOf(counts[a][p]).length());
            }
        }
        cellWidth += 2;

        StringBuilder sb = new StringBuilder();
        sb.append("confusion matrix (rows = actual, columns = predicted)\n");

        sb.append(String.format("%-" + labelWidth + "s", ""));
        for(int p = 0; p < classCount; p++){
            sb.append(String.format("%" + cellWidth + "d", p));
        }
        sb.append("\n");

        for(int a = 0; a < classCount; a++){
            sb.append(String.format("%-" + labelWidth + "s", labels[a]));
            for(int p = 0; p < classCount; p++){
                sb.append(String.format("%" + cellWidth + "d", counts[a][p]));
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public String report(String[] classNames){
        checkNames(classNames);

        int nameWidth = 5;
        for(String name : classNames){
            nameWidth = Math.max(nameWidth, name.length());
        }

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-" + nameWidth + "s  %9s  %6s  %6s  %7s\n", "class", "precision", "recall", "f1", "support"));
        for(int c = 0; c < classCount; c++){
            sb.append(String.format(Locale.US, "%-" + nameWidth + "s  %9.3f  %6.3f  %6.3f  %7d\n",
                classNames[c], precision(c), recall(c), f1(c), support(c)));
        }
        sb.append(String.format(Locale.US, "accuracy %.4f   macro f1 %.4f   samples %d\n", accuracy(), macroF1(), total));
        return sb.toString();
    }

    private void checkClass(int c){
        if(c < 0 || c >= classCount){
            throw new IllegalArgumentException("class " + c + " is outside 0.." + (classCount - 1));
        }
    }

    private void checkNames(String[] classNames){
        if(classNames.length != classCount){
            throw new IllegalArgumentException("got " + classNames.length + " class names for " + classCount + " classes");
        }
    }
}
