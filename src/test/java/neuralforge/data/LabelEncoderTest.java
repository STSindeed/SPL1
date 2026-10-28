package neuralforge.data;

import java.util.Arrays;
import static neuralforge.TestHelper.*;

public class LabelEncoderTest{

    public static void main(String[] args){
        testTextLabels();
        testNumericLabels();
        testErrors();
        finish();
    }

    private static void testTextLabels(){
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(new String[]{"versicolor", "setosa", "virginica", "setosa", "versicolor"});

        check("class count", encoder.classCount() == 3);

        check("sorted classes", Arrays.equals(encoder.getClasses(), new String[]{"setosa", "versicolor", "virginica"}));
        check("encode", encoder.encode("setosa") == 0 && encoder.encode("virginica") == 2);
        check("decode", encoder.decode(1).equals("versicolor"));

        int[] all = encoder.encodeAll(new String[]{"virginica", "setosa", "versicolor"});
        check("encode all", Arrays.equals(all, new int[]{2, 0, 1}));

        LabelEncoder other = new LabelEncoder();
        other.fit(new String[]{"virginica", "setosa", "versicolor"});
        check("order does not matter", other.encode("setosa") == 0 && other.encode("versicolor") == 1);

        encoder.getClasses()[0] = "changed";
        check("classes are a copy", encoder.decode(0).equals("setosa"));
    }

    private static void testNumericLabels(){
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(new String[]{"10", "2", "1", "2"});

        check("numeric order", Arrays.equals(encoder.getClasses(), new String[]{"1", "2", "10"}));
        check("numeric encode", encoder.encode("10") == 2);

        LabelEncoder mixed = new LabelEncoder();
        mixed.fit(new String[]{"10", "2", "x"});
        check("mixed labels use text order", Arrays.equals(mixed.getClasses(), new String[]{"10", "2", "x"}));
    }

    private static void testErrors(){
        LabelEncoder encoder = new LabelEncoder();
        expectThrows("encode before fit", () -> encoder.encode("a"));
        expectThrows("count before fit", () -> encoder.classCount());
        expectThrows("fit without labels", () -> encoder.fit(new String[0]));

        encoder.fit(new String[]{"a", "b"});
        expectThrows("unknown label", () -> encoder.encode("c"));
        expectThrows("decode too big", () -> encoder.decode(2));
        expectThrows("decode negative", () -> encoder.decode(-1));
    }
}
