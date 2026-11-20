package neuralforge.data;

import neuralforge.math.Matrix;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

public final class DataPipeline{

    private static final int MIN_ROWS = 10;

    private static final int MAX_FORCED_CLASSES = 100;

    private static final int RARE_CLASS = 5;

    private DataPipeline(){
    }

    public static PreparedData loadByName(String path, String targetName, DataOptions options){
        CsvTable table = CsvReader.readAuto(path, true);
        return prepare(table, table.columnIndex(targetName), options);
    }

    public static PreparedData loadByIndex(String path, boolean hasHeader, int targetColumn, DataOptions options){
        CsvTable table = CsvReader.readAuto(path, hasHeader);
        return prepare(table, targetColumn, options);
    }

    public static PreparedData prepare(CsvTable table, int targetColumn, DataOptions options){
        DataReport report = new DataReport();
        String[] header = table.getHeader();
        checkHeader(header);

        int columns = header.length;
        if(columns < 2){
            throw new DataFormatException("the file has only one column, but a target column and at least one feature column are needed. "
                + "If the values are separated by something unusual, check the delimiter");
        }
        int target = targetColumn < 0 ? columns + targetColumn : targetColumn;
        if(target < 0 || target >= columns){
            throw new DataFormatException("target column " + targetColumn + " does not exist, the file has " + columns + " columns");
        }
        String targetName = header[target];
        report.info("read " + table.rowCount() + " rows and " + columns + " columns, the target is '" + targetName + "'");

        int[] valid = rowsWithLabel(table, target, options, report);
        if(valid.length < MIN_ROWS){
            throw new DataFormatException("only " + valid.length + " usable row(s), at least " + MIN_ROWS + " are needed to train and test a model");
        }

        ColumnProfile[] profiles = TableProfiler.profile(table, valid, options);
        ColumnProfile targetProfile = profiles[target];

        TaskDetector.Detection detection = TaskDetector.detect(targetProfile, options.getMaxClasses());
        TaskType task = detection.getTask();
        String reason = detection.getReason();
        if(task == TaskType.REGRESSION){
            if(!options.isForcingClassification()){
                throw new DataFormatException("the target '" + targetName + "' looks like a regression target: " + reason + ". NeuralForge "
                    + "predicts classes, not numbers. If these numbers really are class codes, switch on forceClassification");
            }
            if(targetProfile.getDistinctCount() > MAX_FORCED_CLASSES){
                throw new DataFormatException("the target '" + targetName + "' has " + targetProfile.getDistinctCount()
                    + " distinct values, too many to be classes even with forceClassification (the limit is " + MAX_FORCED_CLASSES + ")");
            }
            task = TaskType.MULTICLASS_CLASSIFICATION;
            reason = "forced by the user, " + reason;
        }
        report.info("task: " + task.label() + ", " + reason);

        boolean numericTarget = targetProfile.getType() == ColumnType.NUMERIC;
        if(numericTarget && targetProfile.getUnparsableCount() > 0){
            throw DataFormatException.at(table.lineNumber(targetProfile.getFirstBadRow()), targetName, "'" + targetProfile.getFirstBadText()
                + "' is not a number, but most of the target column is. The labels must be consistent");
        }
        int n = valid.length;
        String[] labelTexts = new String[n];
        for(int i = 0; i < n; i++){
            labelTexts[i] = labelText(table.get(valid[i], target), numericTarget, options);
        }
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(labelTexts);
        int[] labels = encoder.encodeAll(labelTexts);
        String[] classNames = encoder.getClasses();
        checkClasses(labels, classNames, report);

        Matrix positions = new Matrix(n, 1);
        for(int i = 0; i < n; i++){
            positions.set(i, 0, i);
        }
        Dataset shell = new Dataset(positions, labels, classNames);
        Dataset[] parts;
        try{
            parts = shell.stratifiedSplit(options.getTestFraction(), new Random(options.getSeed()));
        }catch(IllegalArgumentException e){
            throw new DataFormatException("the " + n + " usable rows cannot be split into training and test parts: " + e.getMessage());
        }
        int[] trainRows = tableRows(parts[0], valid);
        int[] testRows = tableRows(parts[1], valid);
        report.info(String.format(Locale.US, "split into %d training and %d test rows (test fraction %.2f, seed %d)",
            trainRows.length, testRows.length, options.getTestFraction(), options.getSeed()));

        TablePreprocessor preprocessor = new TablePreprocessor(options);
        preprocessor.fit(table, trainRows, target, profiles, report);
        Matrix trainFeatures = preprocessor.transform(table, trainRows, report, "training rows");
        Matrix testFeatures = preprocessor.transform(table, testRows, report, "test rows");

        Dataset train = new Dataset(trainFeatures, parts[0].getLabels(), classNames);
        Dataset test = new Dataset(testFeatures, parts[1].getLabels(), classNames);
        return new PreparedData(train, test, preprocessor, task, reason, numericTarget, targetName, profiles, report);
    }

    private static void checkHeader(String[] header){
        Map<String, Integer> seen = new HashMap<>();
        for(int c = 0; c < header.length; c++){
            if(header[c].isEmpty()){
                throw new DataFormatException("column " + (c + 1) + " has no name in the header row");
            }
            Integer before = seen.put(header[c], c);
            if(before != null){
                throw new DataFormatException("the header has the column name '" + header[c] + "' twice (columns " + (before + 1) + " and " + (c + 1)
                    + "). Every column needs its own name");
            }
        }
    }

    private static int[] rowsWithLabel(CsvTable table, int target, DataOptions options, DataReport report){
        int[] buffer = new int[table.rowCount()];
        int count = 0;
        int firstDroppedLine = -1;
        for(int r = 0; r < table.rowCount(); r++){
            if(options.isMissing(table.get(r, target))){
                if(firstDroppedLine < 0){
                    firstDroppedLine = table.lineNumber(r);
                }
            }else{
                buffer[count++] = r;
            }
        }
        int dropped = table.rowCount() - count;
        if(dropped > 0){
            report.warn(dropped + " row(s) have no label and were dropped (the first is on line " + firstDroppedLine + ")");
        }
        int[] rows = new int[count];
        System.arraycopy(buffer, 0, rows, 0, count);
        return rows;
    }

    public static String labelText(String text, boolean numericTarget, DataOptions options){
        if(numericTarget){
            double value = NumberText.parse(text);
            if(!Double.isNaN(value) && NumberText.isWhole(value)){
                return Long.toString((long) value);
            }
        }
        return options.normalize(text);
    }

    private static void checkClasses(int[] labels, String[] classNames, DataReport report){
        int[] counts = new int[classNames.length];
        for(int label : labels){
            counts[label]++;
        }
        for(int c = 0; c < counts.length; c++){
            if(counts[c] == 1){
                report.warn("class '" + classNames[c] + "' has only one row, it can only be in the training part and cannot be tested");
            }else if(counts[c] < RARE_CLASS){
                report.warn("class '" + classNames[c] + "' has only " + counts[c] + " rows, results for it will be unreliable");
            }
            double share = (double) counts[c] / labels.length;
            if(share > 0.9 && classNames.length > 1){
                report.warn(String.format(Locale.US, "class '%s' makes up %.1f%% of the rows. A model that always answers '%s' would already look "
                    + "accurate, so look at the confusion matrix and not only at the accuracy", classNames[c], 100 * share, classNames[c]));
            }
        }
    }

    private static int[] tableRows(Dataset part, int[] valid){
        int[] rows = new int[part.size()];
        for(int i = 0; i < rows.length; i++){
            rows[i] = valid[(int) part.getFeatures().get(i, 0)];
        }
        return rows;
    }
}
