package neuralforge.data;

import neuralforge.TestData;
import static neuralforge.TestData.bytes;
import static neuralforge.TestHelper.*;

public class DatasetLoaderTest{

    public static void main(String[] args){
        testCsvBasics();
        testLabelColumns();
        testMissingValues();
        testCsvErrors();
        testIdx();
        testIdxErrors();
        finish();
    }

    private static String csv(String content){
        return TestData.writeText(".csv", content);
    }

    private static void testCsvBasics(){
        Dataset data = DatasetLoader.fromCsv(csv("a,b,label\n1,2,x\n3,4,y\n5,6,x\n"), true, -1);

        check("size", data.size() == 3);
        check("feature count", data.featureCount() == 2);
        check("class count", data.classCount() == 2);
        check("class names", data.className(0).equals("x") && data.className(1).equals("y"));
        check("labels", data.getLabels()[0] == 0 && data.getLabels()[1] == 1 && data.getLabels()[2] == 0);
        check("features", data.getFeatures().get(1, 0) == 3 && data.getFeatures().get(2, 1) == 6);

        Dataset noHeader = DatasetLoader.fromCsv(csv("1,2,x\n3,4,y\n"), false, -1);
        check("file without header", noHeader.size() == 2 && noHeader.featureCount() == 2);
    }

    private static void testLabelColumns(){
        String path = csv("label,a,b\nx,1,2\ny,3,4\n");

        Dataset first = DatasetLoader.fromCsv(path, true, 0);
        check("label in the first column", first.getFeatures().get(0, 0) == 1 && first.getFeatures().get(1, 1) == 4);

        Dataset byName = DatasetLoader.fromCsv(path, "label");
        check("label by name", byName.getFeatures().get(0, 1) == 2 && byName.getLabels()[1] == 1);

        Dataset middle = DatasetLoader.fromCsv(csv("a,label,b\n1,x,2\n3,y,4\n"), true, 1);
        check("label in the middle", middle.getFeatures().get(0, 0) == 1 && middle.getFeatures().get(0, 1) == 2);

        String last = csv("a,b,label\n1,2,x\n3,4,y\n");
        Dataset negative = DatasetLoader.fromCsv(last, true, -1);
        Dataset positive = DatasetLoader.fromCsv(last, true, 2);
        check("negative index", negative.getFeatures().approxEquals(positive.getFeatures(), 0.0));
    }

    private static void testMissingValues(){
        Dataset data = DatasetLoader.fromCsv(csv("a,b,label\n1,?,x\n,4,y\n5,6,x\n"), true, -1);

        check("first hole filled", data.getFeatures().get(1, 0) == 3.0);
        check("second hole filled", data.getFeatures().get(0, 1) == 5.0);
        check("other values stay", data.getFeatures().get(0, 0) == 1.0 && data.getFeatures().get(2, 1) == 6.0);

        Dataset words = DatasetLoader.fromCsv(csv("a,label\n2,x\nNA,y\nnan,x\n4,y\n"), true, -1);
        check("NA and nan are missing", words.getFeatures().get(1, 0) == 3.0 && words.getFeatures().get(2, 0) == 3.0);
    }

    private static void testCsvErrors(){
        String message = TestData.messageOf(() -> DatasetLoader.fromCsv(csv("a,b,label\n1,2,x\n3,abc,y\n"), true, -1));
        check("bad number names line and column", message != null && message.contains("line 3") && message.contains("'b'") && message.contains("abc"));

        expectThrows("missing label", () -> DatasetLoader.fromCsv(csv("a,b,label\n1,2,\n"), true, -1));
        expectThrows("column without any value", () -> DatasetLoader.fromCsv(csv("a,b,label\n1,,x\n2,?,y\n"), true, -1));
        expectThrows("only a header", () -> DatasetLoader.fromCsv(csv("a,b,label\n"), true, -1));
        expectThrows("single column", () -> DatasetLoader.fromCsv(csv("a\n1\n2\n"), true, 0));
        expectThrows("label column too big", () -> DatasetLoader.fromCsv(csv("a,b,label\n1,2,x\n"), true, 5));
        expectThrows("label column too small", () -> DatasetLoader.fromCsv(csv("a,b,label\n1,2,x\n"), true, -4));
        expectThrows("unknown label name", () -> DatasetLoader.fromCsv(csv("a,b,label\n1,2,x\n"), "klass"));
        expectThrows("infinite value", () -> DatasetLoader.fromCsv(csv("a,b,label\n1,Infinity,x\n"), true, -1));
    }

    private static String imageFile(){
        byte[] pixels = bytes(0, 255, 0, 255, 128, 128, 128, 128, 255, 255, 255, 255);
        return TestData.writeBytes(".idx", TestData.idxBytes(new int[]{3, 2, 2}, pixels));
    }

    private static String labelFile(int... labels){
        return TestData.writeBytes(".idx", TestData.idxBytes(new int[]{labels.length}, bytes(labels)));
    }

    private static void testIdx(){
        Dataset data = DatasetLoader.fromIdx(imageFile(), labelFile(0, 1, 2), 0, 10);

        check("size", data.size() == 3);
        check("one column per pixel", data.featureCount() == 4);
        check("class count is the one we gave", data.classCount() == 10);
        check("digit names", data.className(7).equals("7"));
        check("pixel 0 is 0", data.getFeatures().get(0, 0) == 0.0);
        check("pixel 255 is 1", data.getFeatures().get(0, 1) == 1.0);
        check("labels", data.getLabels()[2] == 2);

        Dataset limited = DatasetLoader.fromIdx(imageFile(), labelFile(0, 1, 2), 2, 10);
        check("limit", limited.size() == 2 && limited.getLabels().length == 2);
    }

    private static void testIdxErrors(){
        expectThrows("number of images and labels differ", () -> DatasetLoader.fromIdx(imageFile(), labelFile(0, 1), 0, 10));
        expectThrows("label outside the classes", () -> DatasetLoader.fromIdx(imageFile(), labelFile(0, 1, 5), 0, 3));
    }
}
