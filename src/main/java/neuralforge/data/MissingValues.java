package neuralforge.data;

import java.util.Locale;
import java.util.Set;

public final class MissingValues{

    private static final String[] DEFAULT_TOKENS = {"", "?", "na", "n/a", "nan", "null"};

    private MissingValues(){
    }

    public static boolean isMissing(String text){
        return isMissing(text, null);
    }

    public static boolean isMissing(String text, Set<String> extraTokens){
        String lower = text.trim().toLowerCase(Locale.ROOT);
        for(String token : DEFAULT_TOKENS){
            if(lower.equals(token)){
                return true;
            }
        }
        return extraTokens != null && extraTokens.contains(lower);
    }
}
