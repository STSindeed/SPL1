package neuralforge.data;

import neuralforge.TestData;
import neuralforge.math.Matrix;
import static neuralforge.TestHelper.*;

public class TablePreprocessorTest{

    private static final String CSV = "x,y,color,konst,id,junk,label\n"
        + "1,10,Red,k,1,j1,a\n"
        + "2,20,red,k,2,j2,b\n"
        + "3,30,Blue,k,3,j3,a\n"
        + "?,40,blue,k,4,j4,b\n"
        + "5,50,?,k,5,j5,a\n"
        + "6,60,green,k,6,j6,b\n"
        + "1000,70,purple,k,7,j7,a\n"
        + ",80,Red,k,8,j8,b\n";

    private static final int[] TRAIN = {0, 1, 2, 3, 4, 5};
    private static final int[] TEST = {6, 7};
    private static final int TARGET = 6;

    public static void main(String[] args){
        testDroppedColumns();
        testFeatureNames();
        testFillUsesTrainingRowsOnly();
        testScaling();
        testOneHot();
        testUnseenCategory();
        testMissingIndicators();
        testCaseSensitive();
        testBadNumbers();
        testColumnsFoundByName();
        testMissingColumnInNewData();
        testLimits();
        testOnlyCategorical();
        testNotFitted();
        finish();
    }

    private static CsvTable table(String csv){
        return CsvReader.read(TestData.writeText(".csv", csv), true);
    }

    private static TablePreprocessor fit(CsvTable table, int[] rows, DataOptions options, DataReport report){
        ColumnProfile[] profiles = TableProfiler.profile(table, null, options);
        TablePreprocessor preprocessor = new TablePreprocessor(options);
        preprocessor.fit(table, rows, TARGET, profiles, report);
        return preprocessor;
    }

    private static String messageOf(Runnable code){
        try{
            code.run();
        }catch(RuntimeException e){
            return e.getMessage();
        }
        return null;
    }

    private static boolean reportContains(DataReport report, String text){
        for(String message : report.getMessages()){
            if(message.contains(text)){
                return true;
            }
        }
        return false;
    }

    private static DataOptions options(){
        return new DataOptions().maxCategories(5);
    }

    private static void testDroppedColumns(){
        DataReport report = new DataReport();
        TablePreprocessor p = fit(table(CSV), TRAIN, options(), report);

        check("three columns dropped", p.getDroppedColumns().size() == 3);
        String dropped = p.getDroppedColumns().toString();
        check("constant column, got " + dropped, dropped.contains("konst: it has the same value in every row"));
        check("identifier column", dropped.contains("id: it looks like a row identifier"));
        check("text identifier", dropped.contains("junk: almost every value is different"));
        check("the report tells about it", reportContains(report, "'konst' was dropped") && report.warningCount() >= 3);

        check("numeric columns", p.getNumericColumns().toString().equals("[x, y]"));
        check("categorical columns", p.getCategoricalColumns().toString().equals("[color]"));
    }

    private static void testFeatureNames(){
        TablePreprocessor p = fit(table(CSV), TRAIN, options(), new DataReport());
        String names = java.util.Arrays.toString(p.getFeatureNames());

        check("names, got " + names, names.equals("[x, y, color=(missing), color=blue, color=green, color=red]"));
        check("feature count", p.featureCount() == 6);
        check("the target is not a feature", !names.contains("label"));
    }

    private static void testFillUsesTrainingRowsOnly(){
        CsvTable t = table(CSV);
        DataReport report = new DataReport();
        TablePreprocessor p = fit(t, TRAIN, options().scale(false), report);
        Matrix train = p.transform(t, TRAIN, report, "training rows");
        Matrix test = p.transform(t, TEST, report, "test rows");

        check("training hole filled with the training median", train.get(3, 0) == 3.0);
        check("test hole filled with the same number", test.get(1, 0) == 3.0);
        check("without scaling the values stay", test.get(0, 0) == 1000.0 && train.get(0, 0) == 1.0);
        check("the report mentions the fill", reportContains(report, "test rows: 1 missing value(s) in 'x' replaced by median 3"));

        DataReport meanReport = new DataReport();
        TablePreprocessor mean = fit(t, TRAIN, options().scale(false).imputation(neuralforge.preprocess.Imputer.Strategy.MEAN), meanReport);
        check("mean of 1, 2, 3, 5, 6", near(mean.transform(t, TEST, meanReport, "test rows").get(1, 0), 3.4, 1e-12));
    }

    private static void testScaling(){
        CsvTable t = table(CSV);
        TablePreprocessor p = fit(t, TRAIN, options(), new DataReport());
        Matrix train = p.transform(t, TRAIN, null, "training rows");
        Matrix test = p.transform(t, TEST, null, "test rows");

        double sum = 0.0;
        double squares = 0.0;
        for(int i = 0; i < train.rows(); i++){
            sum += train.get(i, 1);
            squares += train.get(i, 1) * train.get(i, 1);
        }
        check("training mean is 0", near(sum / train.rows(), 0.0, 1e-9));
        check("training std is 1", near(Math.sqrt(squares / train.rows()), 1.0, 1e-9));

        double std = Math.sqrt((25 * 25 + 15 * 15 + 5 * 5) * 2.0 / 6.0);
        check("test row is scaled with the training numbers", near(test.get(0, 1), (70 - 35) / std, 1e-9));

        Matrix unscaled = fit(t, TRAIN, options().scale(false), new DataReport()).transform(t, TRAIN, null, "training rows");
        check("scaling can be switched off", unscaled.get(5, 1) == 60.0);
    }

    private static void testOneHot(){
        CsvTable t = table(CSV);
        TablePreprocessor p = fit(t, TRAIN, options(), new DataReport());
        Matrix train = p.transform(t, TRAIN, null, "training rows");

        check("Red and red are the same category", train.get(0, 5) == 1.0 && train.get(1, 5) == 1.0);
        check("Blue and blue are the same category", train.get(2, 3) == 1.0 && train.get(3, 3) == 1.0);
        check("a hole is its own category", train.get(4, 2) == 1.0);
        check("green", train.get(5, 4) == 1.0);

        for(int i = 0; i < train.rows(); i++){
            double sum = train.get(i, 2) + train.get(i, 3) + train.get(i, 4) + train.get(i, 5);
            check("row " + i + " has exactly one category", sum == 1.0);
        }
    }

    private static void testUnseenCategory(){
        CsvTable t = table(CSV);
        DataReport report = new DataReport();
        TablePreprocessor p = fit(t, TRAIN, options(), report);
        Matrix test = p.transform(t, TEST, report, "test rows");

        check("unknown category gives all zeros", test.get(0, 2) + test.get(0, 3) + test.get(0, 4) + test.get(0, 5) == 0.0);
        check("the known one is found", test.get(1, 5) == 1.0);
        check("a warning with an example", reportContains(report, "test rows: 1 value(s) of 'color' were not seen in the training rows (for example 'purple')"));
    }

    private static void testMissingIndicators(){
        CsvTable t = table(CSV);
        DataReport report = new DataReport();
        TablePreprocessor p = fit(t, TRAIN, options().missingIndicators(true), report);

        String names = java.util.Arrays.toString(p.getFeatureNames());
        check("only x had holes in the training rows, got " + names, names.endsWith("color=red, x?missing]") && p.featureCount() == 7);

        Matrix train = p.transform(t, TRAIN, report, "training rows");
        Matrix test = p.transform(t, TEST, report, "test rows");
        check("indicator is 1 where the value was missing", train.get(3, 6) == 1.0 && test.get(1, 6) == 1.0);
        check("indicator is 0 elsewhere", train.get(0, 6) == 0.0 && train.get(5, 6) == 0.0 && test.get(0, 6) == 0.0);
    }

    private static void testCaseSensitive(){
        CsvTable t = table(CSV);

        TablePreprocessor p = fit(t, TRAIN, new DataOptions().maxCategories(7).ignoreCase(false), new DataReport());
        String names = java.util.Arrays.toString(p.getFeatureNames());
        check("Red and red are different now, got " + names, names.contains("color=Red") && names.contains("color=red") && names.contains("color=Blue"));
    }

    private static String typoCsv(){
        StringBuilder sb = new StringBuilder("x,y,label\n");
        for(int i = 0; i < 20; i++){
            sb.append(i == 4 ? "abc" : String.valueOf(i)).append(",").append(i % 3).append(",").append(i % 2 == 0 ? "a" : "b").append("\n");
        }
        return sb.toString();
    }

    private static void testBadNumbers(){
        CsvTable t = table(typoCsv());
        int[] rows = TableProfiler.allRows(20);

        String message = messageOf(() -> {
            ColumnProfile[] profiles = TableProfiler.profile(t, null, new DataOptions());
            new TablePreprocessor(new DataOptions()).fit(t, rows, 2, profiles, new DataReport());
        });
        check("error names line and column, got " + message, message != null && message.contains("line 6, column 'x'") && message.contains("'abc'"));
        check("error says how common numbers are", message.contains("19 of the 19") || message.contains("19 of the"));
        check("error offers a way out", message.contains("badNumbersAsMissing"));

        DataOptions lenient = new DataOptions().badNumbersAsMissing(true);
        DataReport report = new DataReport();
        TablePreprocessor p = new TablePreprocessor(lenient);
        p.fit(t, rows, 2, TableProfiler.profile(t, null, lenient), report);
        Matrix result = p.transform(t, rows, report, "training rows");
        check("the stray text is treated as a hole", !Double.isNaN(result.get(4, 0)));
        check("and reported", reportContains(report, "1 value(s) in 'x' are not numbers"));
    }

    private static void testColumnsFoundByName(){
        CsvTable t = table(CSV);
        TablePreprocessor p = fit(t, TRAIN, options(), new DataReport());
        Matrix expected = p.transform(t, TEST, null, "test rows");

        CsvTable other = table("color,y,whatever,x\npurple,70,1,1000\nRed,80,2,\n");
        Matrix actual = p.transform(other, null, null, "new rows");
        check("same numbers from a table with other column order", actual.approxEquals(expected, 1e-12));
    }

    private static void testMissingColumnInNewData(){
        TablePreprocessor p = fit(table(CSV), TRAIN, options(), new DataReport());
        CsvTable other = table("x,color\n1,red\n2,blue\n");
        String message = messageOf(() -> p.transform(other, null, null, "new rows"));
        check("names the missing column, got " + message, message != null && message.contains("no column named 'y'") && message.contains("[x, color]"));
    }

    private static void testLimits(){

        StringBuilder sb = new StringBuilder("v,m,label\n");
        for(int i = 0; i < 10; i++){
            sb.append(i).append(",").append(i < 6 ? "" : String.valueOf(i)).append(",").append(i % 2 == 0 ? "a" : "b").append("\n");
        }
        CsvTable t = table(sb.toString());
        DataOptions options = new DataOptions();
        TablePreprocessor p = new TablePreprocessor(options);
        p.fit(t, TableProfiler.allRows(10), 2, TableProfiler.profile(t, null, options), new DataReport());
        check("column with too many holes is dropped, got " + p.getDroppedColumns(), p.getDroppedColumns().toString().contains("m: 60.0% of its values are missing"));
        check("v is kept", p.getNumericColumns().toString().equals("[v]"));

        StringBuilder many = new StringBuilder("v,c,label\n");
        for(int i = 0; i < 12; i++){
            many.append(i).append(",c").append(i % 3).append(",").append(i % 2 == 0 ? "a" : "b").append("\n");
        }
        CsvTable t2 = table(many.toString());
        DataOptions strict = new DataOptions().maxCategories(2);
        TablePreprocessor q = new TablePreprocessor(strict);
        q.fit(t2, TableProfiler.allRows(12), 2, TableProfiler.profile(t2, null, strict), new DataReport());
        check("too many categories, got " + q.getDroppedColumns(), q.getDroppedColumns().toString().contains("c: it has 3 distinct values, more than the limit of 2"));

        CsvTable t3 = table("k,label\nz,a\nz,b\nz,a\n");
        DataOptions plain = new DataOptions();
        String message = messageOf(() -> new TablePreprocessor(plain).fit(t3, TableProfiler.allRows(3), 1, TableProfiler.profile(t3, null, plain), new DataReport()));
        check("no usable column, got " + message, message != null && message.contains("no usable feature columns") && message.contains("k:"));
    }

    private static void testOnlyCategorical(){
        CsvTable t = table("c,label\nred,a\nblue,b\nred,a\ngreen,b\nblue,a\nred,b\n");
        int[] rows = TableProfiler.allRows(6);
        DataOptions options = new DataOptions();
        TablePreprocessor p = new TablePreprocessor(options);
        p.fit(t, rows, 1, TableProfiler.profile(t, null, options), new DataReport());
        Matrix result = p.transform(t, rows, null, "rows");

        check("three categories give three features", p.featureCount() == 3 && result.cols() == 3 && result.rows() == 6);
        check("every row has one 1", result.get(0, 2) == 1.0 && result.get(1, 0) == 1.0 && result.get(3, 1) == 1.0);
    }

    private static void testNotFitted(){
        TablePreprocessor p = new TablePreprocessor(new DataOptions());
        CsvTable t = table(CSV);
        expectThrows("transform before fit", () -> p.transform(t, null, null, "rows"));
        expectThrows("feature names before fit", () -> p.getFeatureNames());
        expectThrows("feature count before fit", () -> p.featureCount());
    }
}
