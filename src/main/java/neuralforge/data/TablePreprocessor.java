package neuralforge.data;

import neuralforge.math.Matrix;
import neuralforge.preprocess.Imputer;
import neuralforge.preprocess.OneHotEncoder;
import neuralforge.preprocess.StandardScaler;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class TablePreprocessor{

    private static final String MISSING_CATEGORY = "(missing)";

    private final DataOptions options;

    private final List<String> numericNames = new ArrayList<>();
    private final List<String> categoricalNames = new ArrayList<>();
    private final List<OneHotEncoder> encoders = new ArrayList<>();

    private final List<Boolean> categoryHasMissing = new ArrayList<>();

    private boolean[] hasIndicator;

    private Imputer imputer;
    private StandardScaler scaler;

    private final List<String> dropped = new ArrayList<>();
    private String[] featureNames;
    private boolean fitted = false;

    public TablePreprocessor(DataOptions options){
        this.options = options;
    }

    public void fit(CsvTable table, int[] trainRows, int targetIndex, ColumnProfile[] profiles, DataReport report){
        String[] header = table.getHeader();

        numericNames.clear();
        categoricalNames.clear();
        encoders.clear();
        categoryHasMissing.clear();
        dropped.clear();

        for(int c = 0; c < header.length; c++){
            if(c == targetIndex){
                continue;
            }
            ColumnProfile profile = profiles[c];
            String reason = dropReason(profile);
            if(reason != null){
                dropped.add(profile.getName() + ": " + reason);
                report.warn("column '" + profile.getName() + "' was dropped, " + reason);
                continue;
            }
            if(profile.getType() == ColumnType.NUMERIC){
                if(profile.getUnparsableCount() > 0 && !options.isBadNumbersAsMissing()){
                    throw badNumber(table, profile);
                }
                numericNames.add(profile.getName());
            }else{
                categoricalNames.add(profile.getName());
            }
        }
        if(numericNames.isEmpty() && categoricalNames.isEmpty()){
            throw new DataFormatException("no usable feature columns are left. Columns that were dropped: " + dropped);
        }

        int[] numericColumns = columnIndexes(table, numericNames);

        Matrix raw = numericNames.isEmpty() ? null : readNumeric(table, trainRows, numericColumns, null, "training rows");

        hasIndicator = new boolean[numericNames.size()];
        for(int j = 0; j < numericNames.size(); j++){
            int holes = Imputer.countMissing(raw, j);
            hasIndicator[j] = options.hasMissingIndicators() && holes > 0;
            if(holes == raw.rows()){
                throw new DataFormatException("column '" + numericNames.get(j) + "' has no values in the training rows");
            }
        }

        imputer = null;
        scaler = null;
        if(!numericNames.isEmpty()){
            imputer = new Imputer(options.getImputation());
            Matrix filled = imputer.fitTransform(raw);
            if(options.isScaling()){
                scaler = new StandardScaler();
                scaler.fit(filled);
            }
        }

        for(String name : categoricalNames){
            int column = table.columnIndex(name);
            String[] values = new String[trainRows.length];
            boolean anyMissing = false;
            for(int i = 0; i < trainRows.length; i++){
                String text = table.get(trainRows[i], column);
                if(options.isMissing(text)){
                    values[i] = MISSING_CATEGORY;
                    anyMissing = true;
                }else{
                    values[i] = options.normalize(text);
                }
            }
            OneHotEncoder encoder = new OneHotEncoder();
            encoder.fit(values);
            encoders.add(encoder);
            categoryHasMissing.add(anyMissing);
        }

        List<String> names = new ArrayList<>(numericNames);
        for(int k = 0; k < categoricalNames.size(); k++){
            for(String category : encoders.get(k).getCategories()){
                names.add(categoricalNames.get(k) + "=" + category);
            }
        }
        for(int j = 0; j < numericNames.size(); j++){
            if(hasIndicator[j]){
                names.add(numericNames.get(j) + "?missing");
            }
        }
        featureNames = names.toArray(new String[0]);
        fitted = true;

        report.info("using " + numericNames.size() + " numeric and " + categoricalNames.size() + " categorical columns, "
            + featureNames.length + " features after encoding");
    }

    public Matrix transform(CsvTable table, int[] rows, DataReport report, String setName){
        if(!fitted){
            throw new IllegalStateException("call fit() first");
        }
        int[] used = rows != null ? rows : TableProfiler.allRows(table.rowCount());
        int count = used.length;

        int[] numericColumns = columnIndexes(table, numericNames);
        Matrix raw = numericNames.isEmpty() ? null : readNumeric(table, used, numericColumns, report, setName);
        Matrix numeric = null;
        if(!numericNames.isEmpty()){
            numeric = imputer.transform(raw);
            if(scaler != null){
                numeric = scaler.transform(numeric);
            }
        }

        Matrix result = new Matrix(count, featureNames.length);

        for(int j = 0; j < numericNames.size(); j++){
            for(int i = 0; i < count; i++){
                result.set(i, j, numeric.get(i, j));
            }
            int holes = Imputer.countMissing(raw, j);
            if(holes > 0 && report != null){
                report.info(setName + ": " + holes + " missing value(s) in '" + numericNames.get(j) + "' replaced by "
                    + options.getImputation().name().toLowerCase(Locale.ROOT) + " " + String.format(Locale.US, "%.4g", imputer.getFill()[j]));
            }
        }

        int offset = numericNames.size();
        for(int k = 0; k < categoricalNames.size(); k++){
            String name = categoricalNames.get(k);
            int column = columnIndexes(table, List.of(name))[0];
            OneHotEncoder encoder = encoders.get(k);

            int unseen = 0;
            int unexpectedMissing = 0;
            String example = null;
            for(int i = 0; i < count; i++){
                String text = table.get(used[i], column);
                String key;
                if(options.isMissing(text)){

                    key = categoryHasMissing.get(k) ? MISSING_CATEGORY : null;
                    if(key == null){
                        unexpectedMissing++;
                    }
                }else{
                    key = options.normalize(text);
                }
                if(key != null && !encoder.encodeInto(result, i, offset, key)){
                    unseen++;
                    if(example == null){
                        example = text;
                    }
                }
            }
            if(unseen > 0 && report != null){
                report.warn(setName + ": " + unseen + " value(s) of '" + name + "' were not seen in the training rows (for example '"
                    + example + "'), they are encoded as all zeros");
            }
            if(unexpectedMissing > 0 && report != null){
                report.info(setName + ": " + unexpectedMissing + " missing value(s) in '" + name
                    + "' (there were none in the training rows), they are encoded as all zeros");
            }
            offset += encoder.size();
        }

        for(int j = 0; j < numericNames.size(); j++){
            if(hasIndicator[j]){
                for(int i = 0; i < count; i++){
                    result.set(i, offset, Double.isNaN(raw.get(i, j)) ? 1.0 : 0.0);
                }
                offset++;
            }
        }
        return result;
    }

    public String[] getFeatureNames(){
        checkFitted();
        return featureNames.clone();
    }

    public int featureCount(){
        checkFitted();
        return featureNames.length;
    }

    public List<String> getNumericColumns(){
        return new ArrayList<>(numericNames);
    }

    public List<String> getCategoricalColumns(){
        return new ArrayList<>(categoricalNames);
    }

    public List<String> getDroppedColumns(){
        return new ArrayList<>(dropped);
    }

    private String dropReason(ColumnProfile p){
        if(p.getKnownCount() == 0){
            return "it has no values";
        }
        if(p.getMissingFraction() > options.getMaxMissingFraction()){
            return String.format(Locale.US, "%.1f%% of its values are missing (the limit is %.1f%%)",
                100 * p.getMissingFraction(), 100 * options.getMaxMissingFraction());
        }
        if(p.getDistinctCount() <= 1){
            return "it has the same value in every row";
        }
        if(p.getType() == ColumnType.CATEGORICAL && p.getDistinctCount() > options.getMaxCategories()){

            if(p.getDistinctCount() >= 0.9 * p.getKnownCount()){
                return "almost every value is different (" + p.getDistinctCount() + " distinct values), it looks like an identifier or free text";
            }
            return "it has " + p.getDistinctCount() + " distinct values, more than the limit of " + options.getMaxCategories() + " for one-hot encoding";
        }
        if(p.getType() == ColumnType.NUMERIC && looksLikeIdentifierName(p.getName()) && p.isAllWhole()
            && p.getDistinctCount() == p.getKnownCount()){
            return "it looks like a row identifier (every value is different)";
        }
        return null;
    }

    private static boolean looksLikeIdentifierName(String name){
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.equals("id") || lower.equals("index") || lower.endsWith("_id") || lower.endsWith(" id") || lower.equals("rownumber");
    }

    private Matrix readNumeric(CsvTable table, int[] rows, int[] columns, DataReport report, String setName){
        Matrix raw = new Matrix(rows.length, columns.length);
        int[] badCounts = new int[columns.length];

        for(int j = 0; j < columns.length; j++){
            String name = numericNames.get(j);
            for(int i = 0; i < rows.length; i++){
                String text = table.get(rows[i], columns[j]);
                double value;
                if(options.isMissing(text)){
                    value = Double.NaN;
                }else{
                    value = NumberText.parse(text);
                    if(Double.isNaN(value)){
                        if(!options.isBadNumbersAsMissing()){
                            throw DataFormatException.at(table.lineNumber(rows[i]), name, "'" + text + "' is not a number");
                        }
                        badCounts[j]++;
                    }
                }
                raw.set(i, j, value);
            }
            if(badCounts[j] > 0 && report != null){
                report.warn(setName + ": " + badCounts[j] + " value(s) in '" + name + "' are not numbers and were treated as missing");
            }
        }
        return raw;
    }

    private DataFormatException badNumber(CsvTable table, ColumnProfile p){
        return DataFormatException.at(table.lineNumber(p.getFirstBadRow()), p.getName(), "'" + p.getFirstBadText()
            + "' is not a number, but " + (p.getKnownCount() - p.getUnparsableCount()) + " of the " + p.getKnownCount()
            + " known values of this column are. " + p.getUnparsableCount() + " cell(s) are affected. Fix them, declare the text as a "
            + "missing value, or switch on badNumbersAsMissing");
    }

    private static int[] columnIndexes(CsvTable table, List<String> names){
        int[] indexes = new int[names.size()];
        String[] header = table.getHeader();
        for(int i = 0; i < indexes.length; i++){
            int found = -1;
            for(int c = 0; c < header.length; c++){
                if(header[c].equals(names.get(i))){
                    found = c;
                    break;
                }
            }
            if(found < 0){
                throw new DataFormatException("the data has no column named '" + names.get(i) + "', which the model needs. The columns are "
                    + Arrays.toString(header));
            }
            indexes[i] = found;
        }
        return indexes;
    }

    private void checkFitted(){
        if(!fitted){
            throw new IllegalStateException("call fit() first");
        }
    }
}
