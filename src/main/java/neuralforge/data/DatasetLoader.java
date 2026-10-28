package neuralforge.data;

import neuralforge.math.Matrix;

public class DatasetLoader{

    private static final String[] MISSING = {"", "?", "na", "n/a", "nan", "null"};

    private DatasetLoader(){
    }

    public static Dataset fromCsv(String path, boolean hasHeader, int labelColumn){
        CsvTable table = CsvReader.read(path, hasHeader);
        return build(table, labelColumn);
    }

    public static Dataset fromCsv(String path, String labelName){
        CsvTable table = CsvReader.read(path, true);
        return build(table, table.columnIndex(labelName));
    }

    private static Dataset build(CsvTable table, int labelColumn){
        int columns = table.columnCount();
        int rows = table.rowCount();
        String[] header = table.getHeader();

        if(rows == 0){
            throw new IllegalArgumentException("the file has no data rows");
        }
        if(columns < 2){
            throw new IllegalArgumentException("need at least 2 columns, one label and one feature");
        }

        int labelIndex = labelColumn < 0 ? columns + labelColumn : labelColumn;
        if(labelIndex < 0 || labelIndex >= columns){
            throw new IllegalArgumentException("label column " + labelColumn + " does not exist, the table has " + columns + " columns");
        }

        String[] labelTexts = new String[rows];
        for(int r = 0; r < rows; r++){
            String text = table.get(r, labelIndex);
            if(isMissing(text)){
                throw new IllegalArgumentException("line " + table.lineNumber(r) + " has no label");
            }
            labelTexts[r] = text;
        }
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(labelTexts);
        int[] labels = encoder.encodeAll(labelTexts);

        int featureCount = columns - 1;
        double[][] values = new double[rows][featureCount];

        for(int r = 0; r < rows; r++){
            int f = 0;
            for(int c = 0; c < columns; c++){
                if(c == labelIndex){
                    continue;
                }
                values[r][f] = parseNumber(table.get(r, c), table.lineNumber(r), header[c]);
                f++;
            }
        }

        int imputed = fillMissing(values, header, labelIndex);
        if(imputed > 0){
            System.out.println("note: " + imputed + " missing values were replaced by the column mean");
        }

        return new Dataset(new Matrix(values), labels, encoder.getClasses());
    }

    private static int fillMissing(double[][] values, String[] header, int labelIndex){
        int rows = values.length;
        int featureCount = values[0].length;
        int replaced = 0;

        for(int f = 0; f < featureCount; f++){
            double sum = 0.0;
            int known = 0;
            for(int r = 0; r < rows; r++){
                if(!Double.isNaN(values[r][f])){
                    sum += values[r][f];
                    known++;
                }
            }

            if(known == rows){
                continue;
            }
            if(known == 0){
                throw new IllegalArgumentException("column '" + featureName(header, labelIndex, f) + "' has no values at all");
            }

            double mean = sum / known;
            for(int r = 0; r < rows; r++){
                if(Double.isNaN(values[r][f])){
                    values[r][f] = mean;
                    replaced++;
                }
            }
        }
        return replaced;
    }

    private static String featureName(String[] header, int labelIndex, int f){
        int column = f < labelIndex ? f : f + 1;
        return header[column];
    }

    private static double parseNumber(String text, int lineNumber, String columnName){
        if(isMissing(text)){
            return Double.NaN;
        }
        double value;
        try{
            value = Double.parseDouble(text);
        }catch(NumberFormatException e){
            throw new IllegalArgumentException("line " + lineNumber + ", column '" + columnName + "': '" + text + "' is not a number");
        }
        if(Double.isInfinite(value)){
            throw new IllegalArgumentException("line " + lineNumber + ", column '" + columnName + "': infinite values are not allowed");
        }
        return value;
    }

    private static boolean isMissing(String text){
        String lower = text.trim().toLowerCase();
        for(String token : MISSING){
            if(lower.equals(token)){
                return true;
            }
        }
        return false;
    }

    public static Dataset fromIdx(String imagesPath, String labelsPath, int limit, int classCount){
        Matrix images = IdxReader.readImages(imagesPath, limit);
        int[] labels = IdxReader.readLabels(labelsPath, limit);

        if(images.rows() != labels.length){
            throw new IllegalArgumentException("there are " + images.rows() + " images but " + labels.length + " labels");
        }

        String[] names = new String[classCount];
        for(int i = 0; i < classCount; i++){
            names[i] = String.valueOf(i);
        }
        return new Dataset(images, labels, names);
    }
}
