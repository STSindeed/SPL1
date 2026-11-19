package neuralforge.data;

import neuralforge.TestData;
import static neuralforge.TestHelper.*;

public class TableProfilerTest{

    public static void main(String[] args){
        testTypes();
        testMissingCounts();
        testStatistics();
        testTyposInNumericColumn();
        testTooManyTextsForNumeric();
        testForcedTypes();
        testIgnoreCase();
        testSameNumberWrittenDifferently();
        testSubsetOfRows();
        testTopValue();
        testDescribe();
        finish();
    }

    private static CsvTable table(String csv){
        return CsvReader.read(TestData.writeText(".csv", csv), true);
    }

    private static void testTypes(){
        CsvTable t = table("n,w,cat\n1,x,a\n2.5,y,b\n3,z,a\n");
        ColumnProfile[] p = TableProfiler.profile(t, null, new DataOptions());
        check("one profile per column", p.length == 3);
        check("numbers are numeric", p[0].getType() == ColumnType.NUMERIC);
        check("text is categorical", p[1].getType() == ColumnType.CATEGORICAL && p[2].getType() == ColumnType.CATEGORICAL);
        check("names and indexes", p[2].getName().equals("cat") && p[2].getIndex() == 2);
        check("row count", p[0].getRowCount() == 3);
    }

    private static void testMissingCounts(){
        CsvTable t = table("a,b\n1,x\n,?\nNA,y\n4,\n");
        ColumnProfile[] p = TableProfiler.profile(t, null, new DataOptions());
        check("missing in a", p[0].getMissingCount() == 2 && p[0].getKnownCount() == 2);
        check("missing in b", p[1].getMissingCount() == 2);
        check("missing fraction", near(p[0].getMissingFraction(), 0.5, 1e-12));
        check("distinct ignores missing", p[0].getDistinctCount() == 2 && p[1].getDistinctCount() == 2);

        CsvTable empty = table("a,b\n,\n?,NA\n");
        ColumnProfile[] q = TableProfiler.profile(empty, null, new DataOptions());
        check("a column with no values", q[0].getKnownCount() == 0 && q[0].getDistinctCount() == 0);
    }

    private static void testStatistics(){
        CsvTable t = table("v\n2\n4\n9\n");
        ColumnProfile p = TableProfiler.profile(t, null, new DataOptions())[0];
        check("min", p.getMin() == 2.0);
        check("max", p.getMax() == 9.0);
        check("mean", near(p.getMean(), 5.0, 1e-12));
        check("whole numbers", p.isAllWhole());

        ColumnProfile d = TableProfiler.profile(table("v\n2\n4.5\n"), null, new DataOptions())[0];
        check("decimals are not whole", !d.isAllWhole());
    }

    private static void testTyposInNumericColumn(){
        StringBuilder sb = new StringBuilder("v,w\n");
        for(int i = 1; i <= 19; i++){
            sb.append(i == 8 ? "oops" : String.valueOf(i)).append(",1\n");
        }
        sb.append("20,1\n");
        CsvTable t = table(sb.toString());
        ColumnProfile p = TableProfiler.profile(t, null, new DataOptions())[0];

        check("still numeric", p.getType() == ColumnType.NUMERIC);
        check("one unparsable cell", p.getUnparsableCount() == 1);
        check("it is remembered", p.getFirstBadText().equals("oops"));
        check("its line number is findable", t.lineNumber(p.getFirstBadRow()) == 9);
    }

    private static void testTooManyTextsForNumeric(){
        CsvTable t = table("v\n1\n2\nabc\ndef\n");
        ColumnProfile p = TableProfiler.profile(t, null, new DataOptions())[0];
        check("mostly text is categorical", p.getType() == ColumnType.CATEGORICAL);
        check("no unparsable count for a text column", p.getUnparsableCount() == 0);
    }

    private static void testForcedTypes(){
        CsvTable t = table("zip,note\n1000,3\n2000,4\n1000,5\n");
        ColumnProfile[] p = TableProfiler.profile(t, null, new DataOptions().categorical("zip").numeric("note"));
        check("zip forced to categorical", p[0].getType() == ColumnType.CATEGORICAL);
        check("zip distinct values", p[0].getDistinctCount() == 2);

        CsvTable words = table("v\nabc\ndef\n");
        ColumnProfile q = TableProfiler.profile(words, null, new DataOptions().numeric("v"))[0];
        check("text forced to numeric", q.getType() == ColumnType.NUMERIC && q.getUnparsableCount() == 2);
    }

    private static void testIgnoreCase(){
        CsvTable t = table("c\nRed\nred\nRED\nblue\n");
        check("case ignored: 2 distinct", TableProfiler.profile(t, null, new DataOptions())[0].getDistinctCount() == 2);
        check("case kept: 4 distinct", TableProfiler.profile(t, null, new DataOptions().ignoreCase(false))[0].getDistinctCount() == 4);
    }

    private static void testSameNumberWrittenDifferently(){
        CsvTable t = table("v\n1\n1.0\n1e0\n2\n2.00\n");
        check("numeric: 2 distinct values", TableProfiler.profile(t, null, new DataOptions())[0].getDistinctCount() == 2);
        check("categorical: 5 distinct texts", TableProfiler.profile(t, null, new DataOptions().categorical("v"))[0].getDistinctCount() == 5);
    }

    private static void testSubsetOfRows(){
        CsvTable t = table("v\n1\n2\n3\n4\n5\n");
        ColumnProfile p = TableProfiler.profile(t, new int[]{0, 4}, new DataOptions())[0];
        check("only the chosen rows", p.getRowCount() == 2 && p.getMin() == 1.0 && p.getMax() == 5.0);
        check("allRows", TableProfiler.allRows(3)[2] == 2 && TableProfiler.allRows(3).length == 3);
    }

    private static void testTopValue(){
        CsvTable t = table("c\nb\na\nb\na\nc\n");
        ColumnProfile p = TableProfiler.profile(t, null, new DataOptions())[0];
        check("top value", p.getTopValue().equals("a") && p.getTopCount() == 2);
    }

    private static void testDescribe(){
        CsvTable t = table("age,city\n30,x\n,y\n50,x\n");
        ColumnProfile[] p = TableProfiler.profile(t, null, new DataOptions());
        String numeric = p[0].describe();
        check("numeric line has name, type and range, got " + numeric, numeric.contains("age") && numeric.contains("numeric") && numeric.contains("min"));
        String text = p[1].describe();
        check("categorical line has the most common value, got " + text, text.contains("categorical") && text.contains("most common 'x'"));
    }
}
