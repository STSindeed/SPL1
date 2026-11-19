package neuralforge.preprocess;

import neuralforge.math.Matrix;
import static neuralforge.TestHelper.*;

public class OneHotEncoderTest{

    public static void main(String[] args){
        testCategories();
        testNumericOrder();
        testEncode();
        testUnknown();
        testErrors();
        finish();
    }

    private static OneHotEncoder fitted(String... values){
        OneHotEncoder encoder = new OneHotEncoder();
        encoder.fit(values);
        return encoder;
    }

    private static void testCategories(){
        OneHotEncoder a = fitted("red", "blue", "red", "green");
        OneHotEncoder b = fitted("green", "red", "blue");
        check("three categories", a.size() == 3);
        check("sorted alphabetically", a.getCategories()[0].equals("blue") && a.getCategories()[1].equals("green") && a.getCategories()[2].equals("red"));
        check("row order does not matter", java.util.Arrays.equals(a.getCategories(), b.getCategories()));
        check("indexOf", a.indexOf("blue") == 0 && a.indexOf("red") == 2);

        String[] copy = a.getCategories();
        copy[0] = "changed";
        check("getCategories returns a copy", a.getCategories()[0].equals("blue"));
    }

    private static void testNumericOrder(){
        OneHotEncoder encoder = fitted("10", "2", "1");
        check("numbers sorted as numbers", encoder.getCategories()[0].equals("1") && encoder.getCategories()[1].equals("2")
            && encoder.getCategories()[2].equals("10"));
    }

    private static void testEncode(){
        OneHotEncoder encoder = fitted("a", "b", "c");
        Matrix m = new Matrix(2, 5);

        check("known category", encoder.encodeInto(m, 0, 1, "b"));
        check("second row", encoder.encodeInto(m, 1, 1, "c"));

        check("b is column 2 of row 0", m.get(0, 2) == 1.0);
        check("exactly one 1 in row 0", m.get(0, 1) + m.get(0, 2) + m.get(0, 3) == 1.0);
        check("c is column 3 of row 1", m.get(1, 3) == 1.0 && m.get(1, 2) == 0.0);
        check("other columns untouched", m.get(0, 0) == 0.0 && m.get(0, 4) == 0.0);
    }

    private static void testUnknown(){
        OneHotEncoder encoder = fitted("a", "b");
        check("unknown category has index -1", encoder.indexOf("zzz") == -1);

        Matrix m = new Matrix(1, 2);
        check("encodeInto says false", !encoder.encodeInto(m, 0, 0, "zzz"));
        check("all columns stay zero", m.get(0, 0) == 0.0 && m.get(0, 1) == 0.0);
    }

    private static void testErrors(){
        OneHotEncoder encoder = new OneHotEncoder();
        expectThrows("size before fit", () -> encoder.size());
        expectThrows("indexOf before fit", () -> encoder.indexOf("a"));
        expectThrows("categories before fit", () -> encoder.getCategories());
        expectThrows("fit without values", () -> new OneHotEncoder().fit(new String[0]));
    }
}
