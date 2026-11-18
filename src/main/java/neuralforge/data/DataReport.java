package neuralforge.data;

import java.util.ArrayList;
import java.util.List;

public class DataReport{

    private final List<String> messages = new ArrayList<>();
    private int warnings = 0;

    public void info(String message){
        messages.add(message);
    }

    public void warn(String message){
        messages.add("warning: " + message);
        warnings++;
    }

    public List<String> getMessages(){
        return new ArrayList<>(messages);
    }

    public int warningCount(){
        return warnings;
    }

    public String format(){
        StringBuilder sb = new StringBuilder();
        for(String message : messages){
            sb.append("  ").append(message).append("\n");
        }
        return sb.toString();
    }
}
