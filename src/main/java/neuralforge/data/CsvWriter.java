package neuralforge.data;

import java.io.IOException;
import java.io.Writer;

public final class CsvWriter{

    private CsvWriter(){
    }

    public static String cell(String text, char delimiter){
        boolean needsQuotes = text.indexOf(delimiter) >= 0 || text.indexOf('"') >= 0 || text.indexOf('\n') >= 0 || text.indexOf('\r') >= 0
            || (!text.isEmpty() && (Character.isWhitespace(text.charAt(0)) || Character.isWhitespace(text.charAt(text.length() - 1))));
        if(!needsQuotes){
            return text;
        }
        return "\"" + text.replace("\"", "\"\"") + "\"";
    }

    public static String format(String[] cells, char delimiter){
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < cells.length; i++){
            if(i > 0){
                sb.append(delimiter);
            }
            sb.append(cell(cells[i], delimiter));
        }
        return sb.toString();
    }

    public static void writeRow(Writer out, String[] cells, char delimiter) throws IOException{
        out.write(format(cells, delimiter));
        out.write("\n");
    }
}
