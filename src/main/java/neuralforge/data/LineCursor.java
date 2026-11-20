package neuralforge.data;

import java.util.Arrays;
import java.util.List;

public class LineCursor{

    private final List<String> lines;
    private final String source;
    private int position = 0;

    public LineCursor(List<String> lines, String source){
        this.lines = lines;
        this.source = source;
    }

    public boolean hasNext(){
        return position < lines.size();
    }

    public String[] next(String tag, int count){
        if(!hasNext()){
            throw new DataFormatException(source + " ended too early, expected a '" + tag + "' line");
        }
        String[] fields;
        try{
            fields = RecordText.fields(lines.get(position));
        }catch(IllegalArgumentException e){
            throw error("damaged text (" + e.getMessage() + ")");
        }
        if(!fields[0].equals(tag)){
            throw error("expected '" + tag + "' but found '" + fields[0] + "'");
        }
        if(fields.length - 1 != count){
            throw error("'" + tag + "' should have " + count + " value(s) but has " + (fields.length - 1));
        }
        position++;
        return Arrays.copyOfRange(fields, 1, fields.length);
    }

    public String nextText(String tag){
        return next(tag, 1)[0];
    }

    public int nextInt(String tag, int min, int max){
        String text = nextText(tag);
        try{
            int value = Integer.parseInt(text);
            if(value < min || value > max){
                position--;
                throw error("'" + tag + "' is " + value + ", it must be between " + min + " and " + max);
            }
            return value;
        }catch(NumberFormatException e){
            position--;
            throw error("'" + tag + "' should be a whole number but is '" + text + "'");
        }
    }

    public boolean nextBoolean(String tag){
        String text = nextText(tag);
        if(text.equals("true")){
            return true;
        }
        if(text.equals("false")){
            return false;
        }
        position--;
        throw error("'" + tag + "' should be true or false but is '" + text + "'");
    }

    public double parseDouble(String text){
        try{
            double value = Double.parseDouble(text);
            if(Double.isNaN(value) || Double.isInfinite(value)){
                throw new NumberFormatException();
            }
            return value;
        }catch(NumberFormatException e){
            throw error("'" + text + "' is not a finite number");
        }
    }

    public DataFormatException error(String message){
        return new DataFormatException(source + ", line " + (position + 1) + ": " + message);
    }
}
