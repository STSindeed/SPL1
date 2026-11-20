package neuralforge.data;

import java.util.ArrayList;
import java.util.List;

public final class RecordText{

    private RecordText(){
    }

    public static String escape(String text){
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < text.length(); i++){
            char c = text.charAt(i);
            switch(c){
                case '\\':
                    sb.append("\\\\");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                default:
                    sb.append(c);
            }
        }
        return sb.toString();
    }

    public static String unescape(String text){
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < text.length(); i++){
            char c = text.charAt(i);
            if(c != '\\'){
                sb.append(c);
                continue;
            }
            if(i + 1 >= text.length()){
                throw new IllegalArgumentException("a backslash at the end of a field");
            }
            char next = text.charAt(++i);
            switch(next){
                case '\\':
                    sb.append('\\');
                    break;
                case 't':
                    sb.append('\t');
                    break;
                case 'n':
                    sb.append('\n');
                    break;
                case 'r':
                    sb.append('\r');
                    break;
                default:
                    throw new IllegalArgumentException("unknown escape \\" + next);
            }
        }
        return sb.toString();
    }

    public static String line(String... fields){
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < fields.length; i++){
            if(i > 0){
                sb.append('\t');
            }
            sb.append(escape(fields[i]));
        }
        return sb.toString();
    }

    public static String[] fields(String line){
        List<String> fields = new ArrayList<>();
        int start = 0;
        for(int i = 0; i <= line.length(); i++){
            if(i == line.length() || line.charAt(i) == '\t'){
                fields.add(unescape(line.substring(start, i)));
                start = i + 1;
            }
        }
        return fields.toArray(new String[0]);
    }
}
