package neuralforge.preprocess;

import neuralforge.math.Matrix;
import static neuralforge.TestHelper.*;

public class ScalerTest{

    public static void main(String[] args){
        testStandardScaler();
        testStandardWithNewData();
        testStandardErrors();
        testMinMaxScaler();
        testMinMaxErrors();
        finish();
    }

    private static Matrix sample(){
        return new Matrix(new double[][]{{1, 10}, {2, 10}, {3, 10}, {4, 10}});
    }

    private static void testStandardScaler(){
        StandardScaler scaler = new StandardScaler();
        Matrix scaled = scaler.fitTransform(sample());

        check("mean", near(scaler.getMean()[0], 2.5, 1e-12));
        check("std", near(scaler.getStd()[0], Math.sqrt(1.25), 1e-12));

        double mean = 0.0;
        double squares = 0.0;
        for(int i = 0; i < 4; i++){
            mean += scaled.get(i, 0) / 4;
            squares += scaled.get(i, 0) * scaled.get(i, 0) / 4;
        }
        check("scaled mean is 0", near(mean, 0.0, 1e-12));
        check("scaled std is 1", near(Math.sqrt(squares), 1.0, 1e-12));

        boolean allZero = true;
        for(int i = 0; i < 4; i++){
            if(scaled.get(i, 1) != 0.0){
                allZero = false;
            }
        }
        check("constant column gives zeros", allZero);

        check("inverse transform", scaler.inverseTransform(scaled).approxEquals(sample(), 1e-12));
        check("name", scaler.name().equals("standard"));
    }

    private static void testStandardWithNewData(){
        StandardScaler scaler = new StandardScaler();
        scaler.fit(sample());

        double std = Math.sqrt(1.25);
        Matrix other = new Matrix(new double[][]{{2.5, 10}, {2.5 + std, 12}});
        Matrix scaled = scaler.transform(other);
        check("new data at the mean", near(scaled.get(0, 0), 0.0, 1e-12));
        check("new data one std above", near(scaled.get(1, 0), 1.0, 1e-12));
        check("value of constant column that changed", near(scaled.get(1, 1), 2.0, 1e-12));

        check("input untouched", other.get(1, 1) == 12);
    }

    private static void testStandardErrors(){
        StandardScaler scaler = new StandardScaler();
        expectThrows("transform before fit", () -> scaler.transform(sample()));
        expectThrows("mean before fit", () -> scaler.getMean());

        scaler.fit(sample());
        expectThrows("wrong column count", () -> scaler.transform(new Matrix(2, 3)));
    }

    private static void testMinMaxScaler(){
        MinMaxScaler scaler = new MinMaxScaler();
        Matrix scaled = scaler.fitTransform(sample());

        check("min goes to 0", near(scaled.get(0, 0), 0.0, 1e-12));
        check("middle value", near(scaled.get(1, 0), 1.0 / 3.0, 1e-12));
        check("max goes to 1", near(scaled.get(3, 0), 1.0, 1e-12));
        check("constant column gives zeros", scaled.get(2, 1) == 0.0);

        Matrix outside = scaler.transform(new Matrix(new double[][]{{5, 10}, {0, 10}}));
        check("above the range", near(outside.get(0, 0), 4.0 / 3.0, 1e-12));
        check("below the range", near(outside.get(1, 0), -1.0 / 3.0, 1e-12));
        check("name", scaler.name().equals("minmax"));
    }

    private static void testMinMaxErrors(){
        MinMaxScaler scaler = new MinMaxScaler();
        expectThrows("transform before fit", () -> scaler.transform(sample()));

        scaler.fit(sample());
        expectThrows("wrong column count", () -> scaler.transform(new Matrix(2, 3)));
    }
}
