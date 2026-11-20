package neuralforge.cli;

import neuralforge.data.CsvTable;
import neuralforge.data.CsvWriter;
import neuralforge.data.DataReport;
import neuralforge.data.TableProfiler;
import neuralforge.io.ModelBundle;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import java.io.IOException;
import java.io.Writer;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

public class Predictor{

    private static final int CHUNK = 1000;

    private final ModelBundle bundle;

    public Predictor(ModelBundle bundle){
        this.bundle = bundle;
    }

    public PredictionResult predict(CsvTable table, int[] rows, DataReport report){
        int[] used = rows != null ? rows : TableProfiler.allRows(table.rowCount());
        if(used.length == 0){
            throw new IllegalArgumentException("there are no rows to predict");
        }

        Matrix features = bundle.getPreprocessor().transform(table, used, report, "the data");
        NeuralNetwork network = bundle.getNetwork();
        int classCount = bundle.getClassNames().length;

        Matrix probabilities = new Matrix(used.length, classCount);
        int[] classes = new int[used.length];
        int[] order = TableProfiler.allRows(used.length);

        for(int start = 0; start < used.length; start += CHUNK){
            int end = Math.min(start + CHUNK, used.length);
            Matrix output = network.forward(features.selectRows(order, start, end));
            int[] best = output.argmaxRows();
            for(int i = 0; i < best.length; i++){
                classes[start + i] = best[i];
                for(int c = 0; c < classCount; c++){
                    probabilities.set(start + i, c, output.get(i, c));
                }
            }
        }
        return new PredictionResult(classes, probabilities, bundle.getClassNames());
    }

    public void writeCsv(CsvTable table, int[] rows, PredictionResult result, boolean withProbabilities, char delimiter, Writer out) throws IOException{
        if(rows.length != result.size()){
            throw new IllegalArgumentException("there are " + rows.length + " rows but " + result.size() + " answers");
        }

        String[] original = table.getHeader();
        String[] classNames = bundle.getClassNames();

        Set<String> taken = new HashSet<>();
        for(String name : original){
            taken.add(name);
        }
        int extra = 2 + (withProbabilities ? classNames.length : 0);
        String[] header = new String[original.length + extra];
        System.arraycopy(original, 0, header, 0, original.length);
        int k = original.length;
        header[k++] = unique("predicted_" + bundle.getTargetName(), taken);
        header[k++] = unique("confidence", taken);
        if(withProbabilities){
            for(String name : classNames){
                header[k++] = unique("prob_" + name, taken);
            }
        }
        CsvWriter.writeRow(out, header, delimiter);

        for(int i = 0; i < rows.length; i++){
            String[] row = new String[header.length];
            for(int c = 0; c < original.length; c++){
                row[c] = table.get(rows[i], c);
            }
            int j = original.length;
            row[j++] = result.className(i);
            row[j++] = String.format(Locale.US, "%.6f", result.confidence(i));
            if(withProbabilities){
                for(int c = 0; c < classNames.length; c++){
                    row[j++] = String.format(Locale.US, "%.6f", result.probability(i, c));
                }
            }
            CsvWriter.writeRow(out, row, delimiter);
        }
    }

    private static String unique(String name, Set<String> taken){
        String candidate = name;
        int n = 2;
        while(!taken.add(candidate)){
            candidate = name + "_" + n++;
        }
        return candidate;
    }
}
