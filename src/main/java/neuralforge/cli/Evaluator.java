package neuralforge.cli;

import neuralforge.data.CsvTable;
import neuralforge.data.DataFormatException;
import neuralforge.data.DataReport;
import neuralforge.io.ModelBundle;
import neuralforge.loss.CrossEntropy;
import neuralforge.math.Matrix;
import neuralforge.metrics.ConfusionMatrix;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Evaluator{

    public static class Result{

        private final int rowCount;
        private final int missingLabels;
        private final int unknownLabels;
        private final String exampleUnknown;
        private final int[] usedRows;
        private final int[] actual;
        private final PredictionResult predictions;
        private final double accuracy;
        private final double loss;
        private final ConfusionMatrix matrix;

        Result(int rowCount, int missingLabels, int unknownLabels, String exampleUnknown, int[] usedRows, int[] actual, PredictionResult predictions,
               double accuracy, double loss, ConfusionMatrix matrix){
            this.rowCount = rowCount;
            this.missingLabels = missingLabels;
            this.unknownLabels = unknownLabels;
            this.exampleUnknown = exampleUnknown;
            this.usedRows = usedRows;
            this.actual = actual;
            this.predictions = predictions;
            this.accuracy = accuracy;
            this.loss = loss;
            this.matrix = matrix;
        }

        public int getRowCount(){
            return rowCount;
        }

        public int getUsedCount(){
            return usedRows.length;
        }

        public int getMissingLabels(){
            return missingLabels;
        }

        public int getUnknownLabels(){
            return unknownLabels;
        }

        public String getExampleUnknown(){
            return exampleUnknown;
        }

        public double getAccuracy(){
            return accuracy;
        }

        public double getLoss(){
            return loss;
        }

        public ConfusionMatrix getMatrix(){
            return matrix;
        }

        public PredictionResult getPredictions(){
            return predictions;
        }

        public int[] getActual(){
            return actual.clone();
        }

        private int[] wrongPositions(){
            List<Integer> wrong = new ArrayList<>();
            for(int i = 0; i < actual.length; i++){
                if(predictions.classOf(i) != actual[i]){
                    wrong.add(i);
                }
            }
            int[] positions = new int[wrong.size()];
            for(int i = 0; i < positions.length; i++){
                positions[i] = wrong.get(i);
            }
            return positions;
        }

        public int wrongCount(){
            return wrongPositions().length;
        }

        public int[] wrongTableRows(){
            int[] positions = wrongPositions();
            int[] rows = new int[positions.length];
            for(int i = 0; i < rows.length; i++){
                rows[i] = usedRows[positions[i]];
            }
            return rows;
        }

        public PredictionResult wrongPredictions(){
            return predictions.select(wrongPositions());
        }
    }

    private Evaluator(){
    }

    public static Result evaluate(ModelBundle bundle, CsvTable table, DataReport report){
        String target = bundle.getTargetName();

        int column = -1;
        String[] header = table.getHeader();
        for(int c = 0; c < header.length; c++){
            if(header[c].equals(target)){
                column = c;
                break;
            }
        }
        if(column < 0){
            throw new DataFormatException("the file has no column '" + target + "', the target the model was trained on. Evaluating needs the "
                + "true labels to compare the answers with. The columns are " + Arrays.toString(table.getHeader()));
        }

        List<Integer> usable = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        int missing = 0;
        int unknown = 0;
        String example = null;
        for(int r = 0; r < table.rowCount(); r++){
            String text = table.get(r, column);
            if(bundle.isMissingLabel(text)){
                missing++;
                continue;
            }
            int index = bundle.classIndexOf(text);
            if(index < 0){
                unknown++;
                if(example == null){
                    example = text;
                }
                continue;
            }
            usable.add(r);
            labels.add(index);
        }
        if(missing > 0){
            report.warn(missing + " row(s) have no label and were left out");
        }
        if(unknown > 0){
            report.warn(unknown + " row(s) have a label the model does not know (for example '" + example + "') and were left out. The model knows "
                + Arrays.toString(bundle.getClassNames()));
        }
        if(usable.isEmpty()){
            throw new DataFormatException("no row has a label that the model knows, so there is nothing to evaluate. The model knows "
                + Arrays.toString(bundle.getClassNames()));
        }

        int n = usable.size();
        int[] rows = new int[n];
        int[] actual = new int[n];
        for(int i = 0; i < n; i++){
            rows[i] = usable.get(i);
            actual[i] = labels.get(i);
        }

        PredictionResult predictions = new Predictor(bundle).predict(table, rows, report);

        int classCount = bundle.getClassNames().length;
        Matrix targets = new Matrix(n, classCount);
        Matrix probabilities = new Matrix(n, classCount);
        for(int i = 0; i < n; i++){
            targets.set(i, actual[i], 1.0);
            for(int c = 0; c < classCount; c++){
                probabilities.set(i, c, predictions.probability(i, c));
            }
        }
        double loss = new CrossEntropy().compute(probabilities, targets);

        ConfusionMatrix matrix = ConfusionMatrix.of(actual, predictions.getClasses(), classCount);
        return new Result(table.rowCount(), missing, unknown, example, rows, actual, predictions, matrix.accuracy(), loss, matrix);
    }
}
