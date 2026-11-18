package neuralforge.data;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class TableProfiler{

    private static final double NUMERIC_SHARE = 0.9;

    private TableProfiler(){
    }

    public static ColumnProfile[] profile(CsvTable table, int[] rows, DataOptions options){
        int[] used = rows != null ? rows : allRows(table.rowCount());
        String[] header = table.getHeader();

        ColumnProfile[] result = new ColumnProfile[table.columnCount()];
        for(int c = 0; c < result.length; c++){
            result[c] = profileColumn(table, used, c, header[c], options);
        }
        return result;
    }

    private static String numberKey(double value){
        if(NumberText.isWhole(value) && Math.abs(value) < 1e15){
            return Long.toString((long) value);
        }
        return Double.toString(value);
    }

    public static int[] allRows(int count){
        int[] rows = new int[count];
        for(int i = 0; i < count; i++){
            rows[i] = i;
        }
        return rows;
    }

    private static ColumnProfile profileColumn(CsvTable table, int[] rows, int column, String name, DataOptions options){

        Map<String, Integer> counts = new HashMap<>();

        Set<String> numericKeys = new HashSet<>();

        int missing = 0;
        int known = 0;
        int numbers = 0;
        double sum = 0.0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        boolean allWhole = true;
        int firstBadRow = -1;
        String firstBadText = null;

        for(int row : rows){
            String text = table.get(row, column);
            if(options.isMissing(text)){
                missing++;
                continue;
            }
            known++;
            counts.merge(options.normalize(text), 1, Integer::sum);

            double value = NumberText.parse(text);
            numericKeys.add(Double.isNaN(value) ? options.normalize(text) : numberKey(value));
            if(Double.isNaN(value)){
                if(firstBadRow < 0){
                    firstBadRow = row;
                    firstBadText = text;
                }
            }else{
                numbers++;
                sum += value;
                min = Math.min(min, value);
                max = Math.max(max, value);
                if(!NumberText.isWhole(value)){
                    allWhole = false;
                }
            }
        }

        ColumnType type;
        if(options.isForcedCategorical(name)){
            type = ColumnType.CATEGORICAL;
        }else if(options.isForcedNumeric(name)){
            type = ColumnType.NUMERIC;
        }else if(known == 0){
            type = ColumnType.CATEGORICAL;
        }else if(numbers == known || numbers >= NUMERIC_SHARE * known){
            type = ColumnType.NUMERIC;
        }else{
            type = ColumnType.CATEGORICAL;
        }

        int unparsable = type == ColumnType.NUMERIC ? known - numbers : 0;

        String topValue = null;
        int topCount = 0;
        for(Map.Entry<String, Integer> entry : counts.entrySet()){

            if(entry.getValue() > topCount || (entry.getValue() == topCount && entry.getKey().compareTo(topValue) < 0)){
                topValue = entry.getKey();
                topCount = entry.getValue();
            }
        }

        double mean = numbers > 0 ? sum / numbers : Double.NaN;
        int distinct = type == ColumnType.NUMERIC ? numericKeys.size() : counts.size();
        return new ColumnProfile(name, column, type, rows.length, missing, distinct, unparsable, firstBadRow, firstBadText,
            min, max, mean, allWhole, topValue, topCount);
    }
}
