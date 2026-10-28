package neuralforge.data;

import neuralforge.math.Matrix;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import static neuralforge.TestHelper.*;

public class DatasetTest{

    public static void main(String[] args){
        testBasics();
        testTargets();
        testSubsetAndWithFeatures();
        testSplitSizes();
        testSplitCoversEverything();
        testSplitReproducible();
        testSplitSmallClasses();
        testErrors();
        finish();
    }

    private static Dataset sample(){
        int[] labels = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2};
        Matrix features = new Matrix(12, 2);
        for(int i = 0; i < 12; i++){
            features.set(i, 0, i);
            features.set(i, 1, 10 * i);
        }
        return new Dataset(features, labels, new String[]{"a", "b", "c"});
    }

    private static int[] countClasses(Dataset data){
        return data.classCounts();
    }

    private static void testBasics(){
        Dataset data = sample();
        check("size", data.size() == 12);
        check("feature count", data.featureCount() == 2);
        check("class count", data.classCount() == 3);
        check("class name", data.className(2).equals("c"));

        int[] counts = data.classCounts();
        check("class counts", counts[0] == 6 && counts[1] == 4 && counts[2] == 2);
        check("describe", data.describe().contains("12 samples") && data.describe().contains("3 classes"));

        data.getLabels()[0] = 2;
        check("labels are a copy", data.getLabels()[0] == 0);
    }

    private static void testTargets(){
        Dataset data = sample();
        Matrix targets = data.getTargets();
        check("targets shape", targets.shape().equals("12x3"));

        boolean ok = true;
        int[] labels = data.getLabels();
        for(int i = 0; i < 12; i++){
            double rowSum = 0.0;
            for(int c = 0; c < 3; c++){
                rowSum += targets.get(i, c);
                double expected = (c == labels[i]) ? 1.0 : 0.0;
                if(targets.get(i, c) != expected){
                    ok = false;
                }
            }
            if(rowSum != 1.0){
                ok = false;
            }
        }
        check("targets are one-hot", ok);
    }

    private static void testSubsetAndWithFeatures(){
        Dataset data = sample();

        Dataset part = data.subset(new int[]{11, 0, 6});
        check("subset size", part.size() == 3);
        check("subset keeps the given order", part.getFeatures().get(0, 0) == 11 && part.getFeatures().get(1, 0) == 0);
        check("subset labels follow the rows", part.getLabels()[0] == 2 && part.getLabels()[2] == 1);
        check("subset keeps the classes", part.classCount() == 3);

        Dataset changed = data.withFeatures(Matrix.zeros(12, 5));
        check("new features", changed.featureCount() == 5);
        check("same labels", changed.getLabels()[11] == 2);
        expectThrows("wrong number of rows", () -> data.withFeatures(Matrix.zeros(11, 2)));
    }

    private static void testSplitSizes(){
        Dataset data = sample();
        Dataset[] parts = data.stratifiedSplit(0.25, new Random(1));

        check("train size", parts[0].size() == 8);
        check("test size", parts[1].size() == 4);

        int[] train = countClasses(parts[0]);
        int[] test = countClasses(parts[1]);
        check("test class counts", test[0] == 2 && test[1] == 1 && test[2] == 1);
        check("train class counts", train[0] == 4 && train[1] == 3 && train[2] == 1);
        check("test has all classes", test[0] > 0 && test[1] > 0 && test[2] > 0);
    }

    private static void testSplitCoversEverything(){
        Dataset[] parts = sample().stratifiedSplit(0.4, new Random(5));

        Set<Integer> seen = new HashSet<>();
        boolean duplicate = false;
        for(Dataset part : parts){
            for(int i = 0; i < part.size(); i++){
                int original = (int) part.getFeatures().get(i, 0);
                if(!seen.add(original)){
                    duplicate = true;
                }
            }
        }
        check("no row used twice", !duplicate);
        check("no row lost", seen.size() == 12);

        boolean labelsOk = true;
        Dataset original = sample();
        for(Dataset part : parts){
            for(int i = 0; i < part.size(); i++){
                int row = (int) part.getFeatures().get(i, 0);
                if(part.getLabels()[i] != original.getLabels()[row]){
                    labelsOk = false;
                }
            }
        }
        check("labels follow their rows", labelsOk);
    }

    private static void testSplitReproducible(){
        Dataset[] first = sample().stratifiedSplit(0.3, new Random(99));
        Dataset[] second = sample().stratifiedSplit(0.3, new Random(99));
        check("same seed gives same split", first[1].getFeatures().approxEquals(second[1].getFeatures(), 0.0));
    }

    private static void testSplitSmallClasses(){
        int[] labels = {0, 0, 0, 0, 1};
        Dataset data = new Dataset(new Matrix(5, 1), labels, new String[]{"a", "b"});
        Dataset[] parts = data.stratifiedSplit(0.5, new Random(3));

        check("single sample stays in train", parts[0].classCounts()[1] == 1);
        check("single sample not in test", parts[1].classCounts()[1] == 0);

        Dataset pair = new Dataset(new Matrix(4, 1), new int[]{0, 0, 1, 1}, new String[]{"a", "b"});
        Dataset[] pairParts = pair.stratifiedSplit(0.01, new Random(3));
        check("pair is split 1 and 1", pairParts[1].classCounts()[0] == 1 && pairParts[0].classCounts()[0] == 1);

        Dataset tiny = new Dataset(new Matrix(2, 1), new int[]{0, 1}, new String[]{"a", "b"});
        expectThrows("nothing to test with", () -> tiny.stratifiedSplit(0.5, new Random(1)));
    }

    private static void testErrors(){
        Dataset data = sample();
        expectThrows("fraction 0", () -> data.stratifiedSplit(0.0, new Random(1)));
        expectThrows("fraction 1", () -> data.stratifiedSplit(1.0, new Random(1)));
        expectThrows("negative fraction", () -> data.stratifiedSplit(-0.5, new Random(1)));
        expectThrows("fraction NaN", () -> data.stratifiedSplit(Double.NaN, new Random(1)));

        expectThrows("rows and labels differ", () -> new Dataset(new Matrix(3, 2), new int[]{0, 1}, new String[]{"a", "b"}));
        expectThrows("label too big", () -> new Dataset(new Matrix(2, 2), new int[]{0, 2}, new String[]{"a", "b"}));
        expectThrows("negative label", () -> new Dataset(new Matrix(2, 2), new int[]{0, -1}, new String[]{"a", "b"}));
        expectThrows("no class names", () -> new Dataset(new Matrix(2, 2), new int[]{0, 0}, new String[0]));
    }
}
