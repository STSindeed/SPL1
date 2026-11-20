package neuralforge.preprocess;

import neuralforge.math.Matrix;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class OneHotEncoder{

    private String[] categories;
    private final Map<String, Integer> indexOf = new HashMap<>();

    public void fit(String[] values){
        if(values.length == 0){
            throw new IllegalArgumentException("there are no values to learn categories from");
        }

        Set<String> unique = new LinkedHashSet<>();
        for(String value : values){
            unique.add(value);
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

        categories = names.toArray(new String[0]);
        indexOf.clear();
        for(int i = 0; i < categories.length; i++){
            indexOf.put(categories[i], i);
        }
    }

    public static OneHotEncoder restore(String[] categories){
        if(categories.length == 0){
            throw new IllegalArgumentException("an encoder needs at least one category");
        }
        OneHotEncoder encoder = new OneHotEncoder();
        encoder.categories = categories.clone();
        for(int i = 0; i < categories.length; i++){
            if(encoder.indexOf.put(categories[i], i) != null){
                throw new IllegalArgumentException("the category '" + categories[i] + "' appears twice");
            }
        }
        return encoder;
    }

    public int indexOf(String value){
        checkFitted();
        Integer index = indexOf.get(value);
        return index == null ? -1 : index;
    }

    public boolean encodeInto(Matrix target, int row, int firstColumn, String value){
        int index = indexOf(value);
        if(index < 0){
            return false;
        }
        target.set(row, firstColumn + index, 1.0);
        return true;
    }

    public int size(){
        checkFitted();
        return categories.length;
    }

    public String[] getCategories(){
        checkFitted();
        return categories.clone();
    }

    private void checkFitted(){
        if(categories == null){
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
