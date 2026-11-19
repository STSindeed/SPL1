package neuralforge.preprocess;

import neuralforge.math.Matrix;
import static neuralforge.TestHelper.*;

public class ImputerTest{

    private static final double NAN = Double.NaN;

    public static void main(String[] args){
        testMean();
        testMedian();
        testEvenMedian();
        testTransform();
        testFitOnTrainOnly();
        testErrors();
        testCountMissing();
        finish();
    }

    private static Matrix data(){
        return new Matrix(new double[][]{
            {1, 10, 100},
            {NAN, 20, 100},
            {3, NAN, 400},
            {8, 30, NAN}
        });
    }

    private static void testMean(){
        Imputer imputer = new Imputer(Imputer.Strategy.MEAN);
        imputer.fit(data());
        double[] fill = imputer.getFill();
        check("mean of column 0 ignores the hole", near(fill[0], 4.0, 1e-12));
        check("mean of column 1", near(fill[1], 20.0, 1e-12));
        check("mean of column 2", near(fill[2], 200.0, 1e-12));
        check("strategy is stored", imputer.getStrategy() == Imputer.Strategy.MEAN);
    }

    private static void testMedian(){
        Imputer imputer = new Imputer(Imputer.Strategy.MEDIAN);
        imputer.fit(data());
        double[] fill = imputer.getFill();
        check("median of column 0 (1, 3, 8)", fill[0] == 3.0);
        check("median of column 1 (10, 20, 30)", fill[1] == 20.0);
        check("median of column 2 (100, 100, 400)", fill[2] == 100.0);
    }

    private static void testEvenMedian(){
        Imputer imputer = new Imputer(Imputer.Strategy.MEDIAN);
        imputer.fit(new Matrix(new double[][]{{1}, {2}, {NAN}, {10}, {20}}));
        check("average of the two middle values (2 and 10)", imputer.getFill()[0] == 6.0);
    }

    private static void testTransform(){
        Imputer imputer = new Imputer(Imputer.Strategy.MEDIAN);
        Matrix original = data();
        Matrix filled = imputer.fitTransform(original);

        check("hole in column 0", filled.get(1, 0) == 3.0);
        check("hole in column 1", filled.get(2, 1) == 20.0);
        check("hole in column 2", filled.get(3, 2) == 100.0);
        check("known values stay", filled.get(0, 0) == 1.0 && filled.get(2, 2) == 400.0 && filled.get(3, 1) == 30.0);
        check("the input is not changed", Double.isNaN(original.get(1, 0)));

        boolean noHoles = true;
        for(int i = 0; i < filled.rows(); i++){
            for(int j = 0; j < filled.cols(); j++){
                noHoles &= !Double.isNaN(filled.get(i, j));
            }
        }
        check("no holes left", noHoles);
    }

    private static void testFitOnTrainOnly(){
        Imputer imputer = new Imputer(Imputer.Strategy.MEAN);
        imputer.fit(new Matrix(new double[][]{{2}, {4}}));

        Matrix test = new Matrix(new double[][]{{NAN}, {1000}});
        Matrix filled = imputer.transform(test);
        check("test hole is filled with the training mean", filled.get(0, 0) == 3.0);
        check("test values do not change the fill", imputer.getFill()[0] == 3.0);
    }

    private static void testErrors(){
        Imputer imputer = new Imputer(Imputer.Strategy.MEAN);
        expectThrows("transform before fit", () -> imputer.transform(data()));
        expectThrows("fill values before fit", () -> imputer.getFill());
        expectThrows("no strategy", () -> new Imputer(null));
        expectThrows("column with nothing known", () -> new Imputer(Imputer.Strategy.MEAN).fit(new Matrix(new double[][]{{1, NAN}, {2, NAN}})));

        imputer.fit(data());
        expectThrows("other number of columns", () -> imputer.transform(new Matrix(2, 2)));
    }

    private static void testCountMissing(){
        check("count in column 0", Imputer.countMissing(data(), 0) == 1);
        check("count in a column without holes", Imputer.countMissing(new Matrix(3, 1), 0) == 0);
    }
}
