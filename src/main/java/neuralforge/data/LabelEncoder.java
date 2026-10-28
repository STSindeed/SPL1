package neuralforge.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class LabelEncoder{

    private String[] classes;
    private final Map<String, Integer> indexOf = new HashMap<>();

    public void fit(String[] labels){
        if(labels.length == 0){
            throw new IllegalArgumentException("there are no labels to fit");
        }

        Set<String> unique = new LinkedHashSet<>();
        for(String label : labels){
            unique.add(label);
        }

        List<String> names = new ArrayList<>(unique);

        if(allNumeric(names)){
            Collections.sort(names, new Comparator<String>(){
                @Override
                public int compare(String a, String b){
                    return Double.compare(Double.parseDouble(a), Double.parseDouble(b));
                }
            });
        }else{
            Collections.sort(names);
        }

        classes = names.toArray(new String[0]);
        indexOf.clear();
        for(int i = 0; i < classes.length; i++){
            indexOf.put(classes[i], i);
        }
    }

    public int encode(String label){
        checkFitted();
        Integer index = indexOf.get(label);
        if(index == null){
            throw new IllegalArgumentException("unknown label '" + label + "'");
        }
        return index;
    }

    public int[] encodeAll(String[] labels){
        int[] result = new int[labels.length];
        for(int i = 0; i < labels.length; i++){
            result[i] = encode(labels[i]);
        }
        return result;
    }

    public String decode(int index){
        checkFitted();
        if(index < 0 || index >= classes.length){
            throw new IllegalArgumentException("class index " + index + " is outside 0.." + (classes.length - 1));
        }
        return classes[index];
    }

    public int classCount(){
        checkFitted();
        return classes.length;
    }

    public String[] getClasses(){
        checkFitted();
        return classes.clone();
    }

    private void checkFitted(){
        if(classes == null){
            throw new IllegalStateException("call fit() first");
        }
    }

    private static boolean allNumeric(List<String> names){
        for(String name : names){
            try{
                Double.parseDouble(name);
            }catch(NumberFormatException e){
                return false;
            }
        }
        return true;
    }
}
