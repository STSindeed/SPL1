package neuralforge.data;

import java.util.Locale;

public class ColumnProfile{

    private final String name;
    private final int index;
    private final ColumnType type;
    private final int rowCount;
    private final int missingCount;
    private final int distinctCount;

    private final int unparsableCount;

    private final int firstBadRow;
    private final String firstBadText;
    private final double min;
    private final double max;
    private final double mean;
    private final boolean allWhole;
    private final String topValue;
    private final int topCount;

    ColumnProfile(String name, int index, ColumnType type, int rowCount, int missingCount, int distinctCount,
                  int unparsableCount, int firstBadRow, String firstBadText, double min, double max, double mean,
                  boolean allWhole, String topValue, int topCount){
        this.name = name;
        this.index = index;
        this.type = type;
        this.rowCount = rowCount;
        this.missingCount = missingCount;
        this.distinctCount = distinctCount;
        this.unparsableCount = unparsableCount;
        this.firstBadRow = firstBadRow;
        this.firstBadText = firstBadText;
        this.min = min;
        this.max = max;
        this.mean = mean;
        this.allWhole = allWhole;
        this.topValue = topValue;
        this.topCount = topCount;
    }

    public String getName(){
        return name;
    }

    public int getIndex(){
        return index;
    }

    public ColumnType getType(){
        return type;
    }

    public int getRowCount(){
        return rowCount;
    }

    public int getMissingCount(){
        return missingCount;
    }

    public int getKnownCount(){
        return rowCount - missingCount;
    }

    public double getMissingFraction(){
        return rowCount == 0 ? 0.0 : (double) missingCount / rowCount;
    }

    public int getDistinctCount(){
        return distinctCount;
    }

    public int getUnparsableCount(){
        return unparsableCount;
    }

    public int getFirstBadRow(){
        return firstBadRow;
    }

    public String getFirstBadText(){
        return firstBadText;
    }

    public double getMin(){
        return min;
    }

    public double getMax(){
        return max;
    }

    public double getMean(){
        return mean;
    }

    public boolean isAllWhole(){
        return allWhole;
    }

    public String getTopValue(){
        return topValue;
    }

    public int getTopCount(){
        return topCount;
    }

    public String describe(){
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US, "%-18s %-12s missing %4d (%5.1f%%)  distinct %5d", name, type.name().toLowerCase(Locale.ROOT),
            missingCount, 100.0 * getMissingFraction(), distinctCount));
        if(type == ColumnType.NUMERIC && distinctCount > 0){
            sb.append(String.format(Locale.US, "  min %.4g  max %.4g  mean %.4g", min, max, mean));
        }else if(topValue != null){
            sb.append("  most common '").append(topValue).append("' (").append(topCount).append(")");
        }
        return sb.toString();
    }
}
