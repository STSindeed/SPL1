package neuralforge.data;

import neuralforge.preprocess.Imputer;
import static neuralforge.TestHelper.*;

public class NumberAndMissingTest{

    public static void main(String[] args){
        testNumbers();
        testNotNumbers();
        testWhole();
        testMissingTokens();
        testExtraTokens();
        testOptions();
        finish();
    }

    private static void testNumbers(){
        check("integer", NumberText.parse("12") == 12.0);
        check("negative decimal", NumberText.parse("-3.5") == -3.5);
        check("plus sign", NumberText.parse("+4") == 4.0);
        check("no leading digit", NumberText.parse(".5") == 0.5);
        check("no trailing digit", NumberText.parse("5.") == 5.0);
        check("exponent", NumberText.parse("1e3") == 1000.0);
        check("capital exponent with sign", near(NumberText.parse("1E-2"), 0.01, 1e-15));
        check("spaces around are ignored", NumberText.parse("  7 ") == 7.0);
        check("zero", NumberText.parse("0") == 0.0);
        check("isNumber true", NumberText.isNumber("3.14"));
    }

    private static void testNotNumbers(){
        String[] bad = {"12f", "1d", "0x1p3", "Infinity", "-Infinity", "NaN", "abc", "", "1,5", "1e999", "--1", "1.2.3", "12abc", "e5", "."};
        for(String text : bad){
            check("'" + text + "' is not a number", Double.isNaN(NumberText.parse(text)) && !NumberText.isNumber(text));
        }
    }

    private static void testWhole(){
        check("3 is whole", NumberText.isWhole(3.0));
        check("3.0 is whole", NumberText.isWhole(NumberText.parse("3.0")));
        check("3e2 is whole", NumberText.isWhole(NumberText.parse("3e2")));
        check("3.5 is not whole", !NumberText.isWhole(3.5));
        check("negative whole", NumberText.isWhole(-4.0));
    }

    private static void testMissingTokens(){
        String[] missing = {"", "?", "NA", "na", "N/A", "NaN", "null", "NULL", "  ?  ", "   "};
        for(String text : missing){
            check("'" + text + "' is missing", MissingValues.isMissing(text));
        }
        String[] present = {"0", "none", "no", "-", "x", "0.0", "false"};
        for(String text : present){
            check("'" + text + "' is a value", !MissingValues.isMissing(text));
        }
    }

    private static void testExtraTokens(){
        DataOptions options = new DataOptions().missingToken("None").missingToken(" - ");
        check("extra token, case ignored", options.isMissing("NONE") && options.isMissing("none"));
        check("extra token is trimmed", options.isMissing("-"));
        check("built-in tokens still work", options.isMissing("?"));
        check("other texts are not affected", !options.isMissing("no"));
        expectThrows("null token", () -> new DataOptions().missingToken(null));
    }

    private static void testOptions(){
        DataOptions options = new DataOptions();
        check("default imputation", options.getImputation() == Imputer.Strategy.MEDIAN);
        check("default indicators", !options.hasMissingIndicators());
        check("default scaling", options.isScaling());
        check("default ignore case", options.isIgnoringCase());
        check("default max missing", options.getMaxMissingFraction() == 0.5);
        check("default max categories", options.getMaxCategories() == 20);
        check("default max classes", options.getMaxClasses() == 20);
        check("default test fraction", options.getTestFraction() == 0.2);

        options.imputation(Imputer.Strategy.MEAN).missingIndicators(true).scale(false).maxCategories(5).maxClasses(7)
            .testFraction(0.3).seed(9).badNumbersAsMissing(true).forceClassification(true).maxMissingFraction(0.8);
        check("settings are stored", options.getImputation() == Imputer.Strategy.MEAN && options.hasMissingIndicators() && !options.isScaling()
            && options.getMaxCategories() == 5 && options.getMaxClasses() == 7 && options.getTestFraction() == 0.3 && options.getSeed() == 9
            && options.isBadNumbersAsMissing() && options.isForcingClassification() && options.getMaxMissingFraction() == 0.8);

        options.categorical("zip").numeric("rating");
        check("forced columns", options.isForcedCategorical("zip") && options.isForcedNumeric("rating")
            && !options.isForcedCategorical("rating") && !options.isForcedNumeric("zip"));

        check("normalize ignores case by default", new DataOptions().normalize("  Red ").equals("red"));
        check("normalize keeps case when asked", new DataOptions().ignoreCase(false).normalize("  Red ").equals("Red"));

        expectThrows("null imputation", () -> new DataOptions().imputation(null));
        expectThrows("max missing 0", () -> new DataOptions().maxMissingFraction(0.0));
        expectThrows("max missing above 1", () -> new DataOptions().maxMissingFraction(1.5));
        expectThrows("max categories 1", () -> new DataOptions().maxCategories(1));
        expectThrows("max classes 1", () -> new DataOptions().maxClasses(1));
        expectThrows("test fraction 0", () -> new DataOptions().testFraction(0.0));
        expectThrows("test fraction 1", () -> new DataOptions().testFraction(1.0));
    }
}
