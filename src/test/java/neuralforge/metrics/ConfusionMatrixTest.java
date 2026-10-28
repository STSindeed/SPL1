package neuralforge.metrics;

import static neuralforge.TestHelper.*;

public class ConfusionMatrixTest{

    public static void main(String[] args){
        testCounts();
        testMetrics();
        testEdgeCases();
        testText();
        testErrors();
        finish();
    }

    private static ConfusionMatrix sample(){
        int[] actual = {0, 0, 0, 1, 1, 2, 2, 2, 2};
        int[] predicted = {0, 0, 1, 1, 1, 2, 2, 0, 2};
        return ConfusionMatrix.of(actual, predicted, 3);
    }

    private static void testCounts(){
        ConfusionMatrix m = sample();
        check("total", m.total() == 9);
        check("diagonal", m.get(0, 0) == 2 && m.get(1, 1) == 2 && m.get(2, 2) == 3);
        check("off diagonal", m.get(0, 1) == 1 && m.get(2, 0) == 1 && m.get(1, 0) == 0);
        check("support", m.support(0) == 3 && m.support(1) == 2 && m.support(2) == 4);
    }

    private static void testMetrics(){
        ConfusionMatrix m = sample();

        check("accuracy", near(m.accuracy(), 7.0 / 9.0, 1e-12));

        check("precision 0", near(m.precision(0), 2.0 / 3.0, 1e-12));
        check("recall 0", near(m.recall(0), 2.0 / 3.0, 1e-12));
        check("f1 0", near(m.f1(0), 2.0 / 3.0, 1e-12));

        check("precision 1", near(m.precision(1), 2.0 / 3.0, 1e-12));
        check("recall 1", near(m.recall(1), 1.0, 1e-12));
        check("f1 1", near(m.f1(1), 0.8, 1e-12));

        check("precision 2", near(m.precision(2), 1.0, 1e-12));
        check("recall 2", near(m.recall(2), 0.75, 1e-12));
        check("f1 2", near(m.f1(2), 2.0 * 0.75 / 1.75, 1e-12));

        double expected = (2.0 / 3.0 + 0.8 + 2.0 * 0.75 / 1.75) / 3.0;
        check("macro f1", near(m.macroF1(), expected, 1e-12));
    }

    private static void testEdgeCases(){
        ConfusionMatrix empty = new ConfusionMatrix(3);
        check("empty accuracy", empty.accuracy() == 0.0);
        check("empty macro f1", empty.macroF1() == 0.0);

        ConfusionMatrix m = ConfusionMatrix.of(new int[]{0, 1, 2}, new int[]{0, 1, 0}, 3);
        check("never predicted", m.precision(2) == 0.0);
        check("f1 without precision", m.f1(2) == 0.0);

        ConfusionMatrix four = ConfusionMatrix.of(new int[]{0, 1}, new int[]{0, 1}, 4);
        check("unused class is left out", near(four.macroF1(), 1.0, 1e-12));
        check("recall of unused class", four.recall(3) == 0.0);
    }

    private static void testText(){
        ConfusionMatrix m = sample();
        String[] names = {"cat", "dog", "bird"};

        String table = m.format(names);
        check("format has the names", table.contains("cat") && table.contains("bird"));
        check("format has the title", table.contains("rows = actual"));

        String report = m.report(names);
        check("report has the names", report.contains("dog"));
        check("report has accuracy", report.contains("accuracy 0.7778"));
        check("report has the sample count", report.contains("samples 9"));

        String digits = ConfusionMatrix.of(new int[]{0, 1}, new int[]{0, 1}, 2).format(new String[]{"0", "1"});
        check("no doubled labels", !digits.contains("0 0 "));
    }

    private static void testErrors(){
        ConfusionMatrix m = new ConfusionMatrix(3);
        expectThrows("class too big", () -> m.add(3, 0));
        expectThrows("negative class", () -> m.add(0, -1));
        expectThrows("different lengths", () -> m.addAll(new int[]{0, 1}, new int[]{0}));
        expectThrows("get outside", () -> m.get(0, 5));
        expectThrows("wrong number of names", () -> m.format(new String[]{"a", "b"}));
        expectThrows("no classes", () -> new ConfusionMatrix(0));
        check("failed add changes nothing", m.total() == 0);
    }
}
