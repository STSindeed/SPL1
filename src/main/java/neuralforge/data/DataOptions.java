package neuralforge.data;

import neuralforge.preprocess.Imputer;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

public class DataOptions{

    private final Set<String> extraMissing = new HashSet<>();

    private final Set<String> categoricalColumns = new HashSet<>();
    private final Set<String> numericColumns = new HashSet<>();

    private Imputer.Strategy imputation = Imputer.Strategy.MEDIAN;

    private boolean missingIndicators = false;
    private boolean scale = true;

    private boolean ignoreCase = true;

    private boolean badNumbersAsMissing = false;

    private boolean forceClassification = false;

    private double maxMissingFraction = 0.5;

    private int maxCategories = 20;

    private int maxClasses = 20;

    private double testFraction = 0.2;
    private long seed = 42;

    public DataOptions missingToken(String token){
        if(token == null){
            throw new IllegalArgumentException("a missing-value token cannot be null");
        }
        extraMissing.add(token.trim().toLowerCase(Locale.ROOT));
        return this;
    }

    public DataOptions categorical(String... columnNames){
        for(String name : columnNames){
            categoricalColumns.add(name);
        }
        return this;
    }

    public DataOptions numeric(String... columnNames){
        for(String name : columnNames){
            numericColumns.add(name);
        }
        return this;
    }

    public DataOptions imputation(Imputer.Strategy strategy){
        if(strategy == null){
            throw new IllegalArgumentException("choose an imputation strategy");
        }
        this.imputation = strategy;
        return this;
    }

    public DataOptions missingIndicators(boolean on){
        this.missingIndicators = on;
        return this;
    }

    public DataOptions scale(boolean on){
        this.scale = on;
        return this;
    }

    public DataOptions ignoreCase(boolean on){
        this.ignoreCase = on;
        return this;
    }

    public DataOptions badNumbersAsMissing(boolean on){
        this.badNumbersAsMissing = on;
        return this;
    }

    public DataOptions forceClassification(boolean on){
        this.forceClassification = on;
        return this;
    }

    public DataOptions maxMissingFraction(double fraction){
        if(!(fraction > 0.0 && fraction <= 1.0)){
            throw new IllegalArgumentException("maxMissingFraction must be in (0, 1], got " + fraction);
        }
        this.maxMissingFraction = fraction;
        return this;
    }

    public DataOptions maxCategories(int count){
        if(count < 2){
            throw new IllegalArgumentException("maxCategories must be at least 2, got " + count);
        }
        this.maxCategories = count;
        return this;
    }

    public DataOptions maxClasses(int count){
        if(count < 2){
            throw new IllegalArgumentException("maxClasses must be at least 2, got " + count);
        }
        this.maxClasses = count;
        return this;
    }

    public DataOptions testFraction(double fraction){
        if(!(fraction > 0.0 && fraction < 1.0)){
            throw new IllegalArgumentException("testFraction must be between 0 and 1, got " + fraction);
        }
        this.testFraction = fraction;
        return this;
    }

    public DataOptions seed(long seed){
        this.seed = seed;
        return this;
    }

    public java.util.List<String> getExtraMissingTokens(){
        java.util.List<String> tokens = new java.util.ArrayList<>(extraMissing);
        java.util.Collections.sort(tokens);
        return tokens;
    }

    public boolean isMissing(String text){
        return MissingValues.isMissing(text, extraMissing);
    }

    public String normalize(String text){
        String trimmed = text.trim();
        return ignoreCase ? trimmed.toLowerCase(Locale.ROOT) : trimmed;
    }

    public boolean isForcedCategorical(String name){
        return categoricalColumns.contains(name);
    }

    public boolean isForcedNumeric(String name){
        return numericColumns.contains(name);
    }

    public Imputer.Strategy getImputation(){
        return imputation;
    }

    public boolean hasMissingIndicators(){
        return missingIndicators;
    }

    public boolean isScaling(){
        return scale;
    }

    public boolean isIgnoringCase(){
        return ignoreCase;
    }

    public boolean isBadNumbersAsMissing(){
        return badNumbersAsMissing;
    }

    public boolean isForcingClassification(){
        return forceClassification;
    }

    public double getMaxMissingFraction(){
        return maxMissingFraction;
    }

    public int getMaxCategories(){
        return maxCategories;
    }

    public int getMaxClasses(){
        return maxClasses;
    }

    public double getTestFraction(){
        return testFraction;
    }

    public long getSeed(){
        return seed;
    }
}
