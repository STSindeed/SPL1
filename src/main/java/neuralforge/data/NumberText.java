package neuralforge.data;

import java.util.regex.Pattern;

public final class NumberText{

    private static final Pattern PLAIN_NUMBER = Pattern.compile("[+-]?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?");

    private NumberText(){
    }

    public static double parse(String text){
        String trimmed = text.trim();
        if(!PLAIN_NUMBER.matcher(trimmed).matches()){
            return Double.NaN;
        }
        double value = Double.parseDouble(trimmed);

        if(Double.isInfinite(value)){
            return Double.NaN;
        }
        return value;
    }

    public static boolean isNumber(String text){
        return !Double.isNaN(parse(text));
    }

    public static boolean isWhole(double value){
        return value == Math.rint(value);
    }
}
