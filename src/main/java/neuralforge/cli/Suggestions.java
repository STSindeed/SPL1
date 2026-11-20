package neuralforge.cli;

import java.util.Collection;
import java.util.Locale;

public final class Suggestions{

    private Suggestions(){
    }

    public static int distance(String a, String b){
        int[] previous = new int[b.length() + 1];
        int[] current = new int[b.length() + 1];

        for(int j = 0; j <= b.length(); j++){
            previous[j] = j;
        }

        for(int i = 1; i <= a.length(); i++){

            current[0] = i;
            for(int j = 1; j <= b.length(); j++){
                int replace = previous[j - 1] + (a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1);
                int delete = previous[j] + 1;
                int insert = current[j - 1] + 1;
                current[j] = Math.min(replace, Math.min(delete, insert));
            }
            int[] swap = previous;
            previous = current;
            current = swap;
        }
        return previous[b.length()];
    }

    public static String closest(String word, Collection<String> candidates){
        String lower = word.toLowerCase(Locale.ROOT);
        int limit = Math.max(2, lower.length() / 3);

        String best = null;
        int bestDistance = Integer.MAX_VALUE;
        for(String candidate : candidates){
            int d = distance(lower, candidate.toLowerCase(Locale.ROOT));

            if(d < bestDistance){
                bestDistance = d;
                best = candidate;
            }
        }
        return bestDistance <= limit ? best : null;
    }

    public static String hint(String word, Collection<String> candidates){
        String best = closest(word, candidates);
        return best == null ? "" : " (did you mean '" + best + "'?)";
    }
}
