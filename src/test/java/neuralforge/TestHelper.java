package neuralforge;

public class TestHelper{

    private static int passed = 0;
    private static int failed = 0;

    public static void check(String name, boolean condition){
        if(condition){
            passed++;
        }else{
            failed++;
            System.out.println("FAIL: " + name);
        }
    }

    public static boolean near(double a, double b, double tolerance){
        return Math.abs(a - b) <= tolerance;
    }

    public static void expectThrows(String name, Runnable code){
        try{
            code.run();
            failed++;
            System.out.println("FAIL (no exception): " + name);
        }catch(RuntimeException e){
            passed++;
        }
    }

    public static void finish(){
        System.out.println();
        System.out.println(passed + " passed, " + failed + " failed");
        if(failed > 0){
            System.exit(1);
        }
    }
}
