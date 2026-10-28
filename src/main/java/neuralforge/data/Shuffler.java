package neuralforge.data;

import java.util.Random;

public class Shuffler{

    private Shuffler(){
    }

    public static int[] range(int n){
        if(n < 0){
            throw new IllegalArgumentException("n cannot be negative, got " + n);
        }
        int[] result = new int[n];
        for(int i = 0; i < n; i++){
            result[i] = i;
        }
        return result;
    }

    public static void shuffle(int[] array, Random rng){
        for(int i = array.length - 1; i > 0; i--){
            int j = rng.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}
