package neuralforge.data;

import neuralforge.TestData;
import static neuralforge.TestHelper.*;

public class TaskDetectorTest{

    public static void main(String[] args){
        testTextTargets();
        testNumericClassCodes();
        testRegression();
        testTwoValues();
        testUnusableTargets();
        testMaxClasses();
        finish();
    }

    private static ColumnProfile profile(String... cells){
        StringBuilder sb = new StringBuilder("t\n");
        for(String cell : cells){
            sb.append(cell).append("\n");
        }
        CsvTable table = CsvReader.read(TestData.writeText(".csv", sb.toString()), true);
        return TableProfiler.profile(table, null, new DataOptions())[0];
    }

    private static TaskDetector.Detection detect(ColumnProfile p){
        return TaskDetector.detect(p, 20);
    }

    private static String messageOf(Runnable code){
        try{
            code.run();
        }catch(RuntimeException e){
            return e.getMessage();
        }
        return null;
    }

    private static void testTextTargets(){
        check("two texts are binary", detect(profile("yes", "no", "yes", "no", "no")).getTask() == TaskType.BINARY_CLASSIFICATION);
        check("three texts are multiclass", detect(profile("a", "b", "c", "a")).getTask() == TaskType.MULTICLASS_CLASSIFICATION);

        TaskDetector.Detection d = detect(profile("a", "b", "c"));
        check("there is always a reason", !d.getReason().isEmpty());
        check("classification flag", d.getTask().isClassification());
    }

    private static void testNumericClassCodes(){
        check("0 and 1 are binary", detect(profile("0", "1", "1", "0")).getTask() == TaskType.BINARY_CLASSIFICATION);
        check("0, 1, 2 are classes", detect(profile("0", "1", "2", "1", "0")).getTask() == TaskType.MULTICLASS_CLASSIFICATION);
        check("whole numbers written with decimals are still codes", detect(profile("1.0", "2.0", "3.0")).getTask() == TaskType.MULTICLASS_CLASSIFICATION);
    }

    private static void testRegression(){
        TaskDetector.Detection decimals = detect(profile("1.5", "2.7", "3.1", "4.9", "0.2"));
        check("decimals are regression", decimals.getTask() == TaskType.REGRESSION);
        check("regression is not classification", !decimals.getTask().isClassification());
        check("the reason shows the range, got " + decimals.getReason(), decimals.getReason().contains("decimals") && decimals.getReason().contains("4.9"));

        String[] many = new String[30];
        for(int i = 0; i < many.length; i++){
            many[i] = String.valueOf(100 + i * 7);
        }
        TaskDetector.Detection whole = detect(profile(many));
        check("many whole numbers are regression", whole.getTask() == TaskType.REGRESSION);
        check("reason mentions whole numbers", whole.getReason().contains("whole numbers"));
    }

    private static void testTwoValues(){
        check("0.5 and 1.5", detect(profile("0.5", "1.5", "0.5")).getTask() == TaskType.BINARY_CLASSIFICATION);
    }

    private static void testUnusableTargets(){
        String same = messageOf(() -> detect(profile("x", "x", "x")));
        check("one value only, got " + same, same != null && same.contains("same value"));

        String none = messageOf(() -> detect(profile("", "?", "NA")));
        check("no values at all, got " + none, none != null && none.contains("no values"));
        check("the error is a DataFormatException", throwsDataFormat(() -> detect(profile("x", "x"))));
    }

    private static boolean throwsDataFormat(Runnable code){
        try{
            code.run();
        }catch(DataFormatException e){
            return true;
        }catch(RuntimeException e){
            return false;
        }
        return false;
    }

    private static void testMaxClasses(){
        String[] names = new String[25];
        for(int i = 0; i < names.length; i++){
            names[i] = "name" + i;
        }
        ColumnProfile text = profile(names);
        String message = messageOf(() -> TaskDetector.detect(text, 20));
        check("too many text values, got " + message, message != null && message.contains("maxClasses") && message.contains("25"));
        check("raising the limit accepts it", TaskDetector.detect(text, 30).getTask() == TaskType.MULTICLASS_CLASSIFICATION);

        String[] codes = new String[25];
        for(int i = 0; i < codes.length; i++){
            codes[i] = String.valueOf(i);
        }
        check("25 whole numbers with limit 20", TaskDetector.detect(profile(codes), 20).getTask() == TaskType.REGRESSION);
        check("25 whole numbers with limit 30", TaskDetector.detect(profile(codes), 30).getTask() == TaskType.MULTICLASS_CLASSIFICATION);
    }
}
