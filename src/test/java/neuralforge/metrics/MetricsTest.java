package neuralforge.metrics;

import neuralforge.math.Matrix;
import static neuralforge.TestHelper.*;

public class MetricsTest{

    public static void main(String[] args){
        testTopK();
        testTies();
        testErrors();
        testMostConfused();
        finish();
    }

    private static Matrix scores(){
        return new Matrix(new double[][]{
            {0.10, 0.60, 0.20, 0.10},
            {0.50, 0.30, 0.15, 0.05},
            {0.40, 0.25, 0.30, 0.05}
        });
    }

    private static void testTopK(){
        int[] labels = {1, 2, 3};
        check("top 1", near(Metrics.topKAccuracy(scores(), labels, 1), 1.0 / 3.0, 1e-12));
        check("top 2", near(Metrics.topKAccuracy(scores(), labels, 2), 1.0 / 3.0, 1e-12));
        check("top 3", near(Metrics.topKAccuracy(scores(), labels, 3), 2.0 / 3.0, 1e-12));
        check("top 4 is always right", near(Metrics.topKAccuracy(scores(), labels, 4), 1.0, 1e-12));
    }

    private static void testTies(){
        Matrix equal = new Matrix(new double[][]{{0.25, 0.25, 0.25, 0.25}});
        check("a tie is decided for the true class", Metrics.topKAccuracy(equal, new int[]{2}, 1) == 1.0);
    }

    private static void testErrors(){
        int[] labels = {1, 2, 3};
        expectThrows("k is 0", () -> Metrics.topKAccuracy(scores(), labels, 0));
        expectThrows("k is too big", () -> Metrics.topKAccuracy(scores(), labels, 5));
        expectThrows("wrong number of labels", () -> Metrics.topKAccuracy(scores(), new int[]{1, 2}, 1));
        expectThrows("label outside the classes", () -> Metrics.topKAccuracy(scores(), new int[]{1, 2, 4}, 1));
        expectThrows("negative label", () -> Metrics.topKAccuracy(scores(), new int[]{1, -1, 3}, 1));
    }

    private static void testMostConfused(){
        ConfusionMatrix m = new ConfusionMatrix(3);
        for(int i = 0; i < 3; i++){
            m.add(0, 1);
        }
        for(int i = 0; i < 2; i++){
            m.add(2, 0);
            m.add(1, 2);
        }
        m.add(1, 1);
        m.add(0, 0);

        int[][] all = m.mostConfused(10);
        check("three kinds of mistakes", all.length == 3);
        check("biggest first", all[0][0] == 0 && all[0][1] == 1 && all[0][2] == 3);

        check("tie goes to the smaller actual class", all[1][0] == 1 && all[1][1] == 2 && all[2][0] == 2);
        check("limit", m.mostConfused(2).length == 2);
        check("limit 0", m.mostConfused(0).length == 0);

        String[] names = {"a", "b", "c"};
        check("text has the pairs", m.formatMostConfused(names, 1).contains("a -> b  3 times"));

        ConfusionMatrix perfect = ConfusionMatrix.of(new int[]{0, 1}, new int[]{0, 1}, 3);
        check("no mistakes", perfect.mostConfused(5).length == 0 && perfect.formatMostConfused(names, 5).contains("no mistakes"));

        expectThrows("negative limit", () -> m.mostConfused(-1));
        expectThrows("wrong number of names", () -> m.formatMostConfused(new String[]{"a"}, 1));
    }
}
