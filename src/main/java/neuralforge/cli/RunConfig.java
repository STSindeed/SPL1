package neuralforge.cli;

import neuralforge.data.DataOptions;
import neuralforge.optimizer.OptimizerFactory;
import neuralforge.preprocess.Imputer;
import neuralforge.schedule.ScheduleFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class RunConfig{

    private interface Setter{
        void set(RunConfig config, ConfigFile.Entry entry);
    }

    private static final Map<String, Setter> SETTERS = new LinkedHashMap<>();

    static{
        SETTERS.put("data.file", (c, e) -> c.dataFile = asText(e));
        SETTERS.put("data.target", (c, e) -> c.target = asText(e));
        SETTERS.put("data.test_fraction", (c, e) -> c.testFraction = asDouble(e, 0.01, 0.99));
        SETTERS.put("data.seed", (c, e) -> c.dataSeed = asLong(e));
        SETTERS.put("data.missing_tokens", (c, e) -> c.missingTokens = asList(e));
        SETTERS.put("data.categorical", (c, e) -> c.categorical = asList(e));
        SETTERS.put("data.numeric", (c, e) -> c.numeric = asList(e));
        SETTERS.put("data.impute", (c, e) -> c.impute = Imputer.Strategy.valueOf(asChoice(e, new String[]{"mean", "median"}).toUpperCase(Locale.ROOT)));
        SETTERS.put("data.missing_indicators", (c, e) -> c.missingIndicators = asBool(e));
        SETTERS.put("data.scale", (c, e) -> c.scale = asBool(e));
        SETTERS.put("data.ignore_case", (c, e) -> c.ignoreCase = asBool(e));
        SETTERS.put("data.max_categories", (c, e) -> c.maxCategories = asInt(e, 2, 1000));
        SETTERS.put("data.max_classes", (c, e) -> c.maxClasses = asInt(e, 2, 1000));
        SETTERS.put("data.max_missing_fraction", (c, e) -> c.maxMissingFraction = asDouble(e, 0.01, 1.0));
        SETTERS.put("data.bad_numbers_as_missing", (c, e) -> c.badNumbersAsMissing = asBool(e));
        SETTERS.put("data.force_classification", (c, e) -> c.forceClassification = asBool(e));

        SETTERS.put("model.hidden", (c, e) -> c.hidden = asIntList(e, 1, 4096, 10));
        SETTERS.put("model.activation", (c, e) -> c.activation = asChoice(e, new String[]{"relu", "leaky_relu", "sigmoid", "tanh"}));
        SETTERS.put("model.init", (c, e) -> c.init = asChoice(e, new String[]{"auto", "he", "xavier"}));
        SETTERS.put("model.dropout", (c, e) -> c.dropout = asDouble(e, 0.0, 0.95));
        SETTERS.put("model.l2", (c, e) -> c.l2 = asDouble(e, 0.0, 100.0));
        SETTERS.put("model.seed", (c, e) -> c.modelSeed = asLong(e));

        SETTERS.put("train.epochs", (c, e) -> c.epochs = asInt(e, 1, 100000));
        SETTERS.put("train.batch_size", (c, e) -> c.batchSize = asInt(e, 1, 100000));
        SETTERS.put("train.optimizer", (c, e) -> c.optimizer = asChoice(e, OptimizerFactory.names()));
        SETTERS.put("train.learning_rate", (c, e) -> c.learningRate = asDouble(e, 1e-9, 100.0));
        SETTERS.put("train.schedule", (c, e) -> c.schedule = asChoice(e, ScheduleFactory.names()));
        SETTERS.put("train.patience", (c, e) -> c.patience = asInt(e, 0, 100000));
        SETTERS.put("train.validation_fraction", (c, e) -> c.validationFraction = asDouble(e, 0.01, 0.9));
        SETTERS.put("train.seed", (c, e) -> c.trainSeed = asLong(e));
        SETTERS.put("train.print_every", (c, e) -> c.printEvery = asInt(e, 0, 100000));

        SETTERS.put("output.model", (c, e) -> c.modelPath = asText(e));
        SETTERS.put("output.history", (c, e) -> c.historyPath = asText(e));
    }

    private String dataFile = null;
    private String target = null;
    private double testFraction = 0.2;
    private long dataSeed = 42;
    private List<String> missingTokens = new ArrayList<>();
    private List<String> categorical = new ArrayList<>();
    private List<String> numeric = new ArrayList<>();
    private Imputer.Strategy impute = Imputer.Strategy.MEDIAN;
    private boolean missingIndicators = false;
    private boolean scale = true;
    private boolean ignoreCase = true;
    private int maxCategories = 20;
    private int maxClasses = 20;
    private double maxMissingFraction = 0.5;
    private boolean badNumbersAsMissing = false;
    private boolean forceClassification = false;

    private List<Integer> hidden = new ArrayList<>(Arrays.asList(16));
    private String activation = "relu";
    private String init = "auto";
    private double dropout = 0.0;
    private double l2 = 0.0;
    private long modelSeed = 2026;

    private int epochs = 100;
    private int batchSize = 32;
    private String optimizer = "adam";

    private double learningRate = Double.NaN;
    private String schedule = "constant";
    private int patience = 0;
    private double validationFraction = 0.15;
    private long trainSeed = 2026;
    private int printEvery = 10;

    private String modelPath = "models/model.nfm";
    private String historyPath = null;

    public static RunConfig fromFile(String path, List<String> overrides){
        ConfigFile file = ConfigFile.read(path);
        for(String assignment : overrides){
            file.override(assignment);
        }
        return fromConfigFile(file);
    }

    public static RunConfig fromText(String text, String sourceName){
        return fromConfigFile(ConfigFile.parse(text, sourceName));
    }

    public static RunConfig fromConfigFile(ConfigFile file){
        RunConfig config = new RunConfig();

        for(ConfigFile.Entry entry : file.entries()){
            Setter setter = SETTERS.get(entry.getKey());
            if(setter == null){
                throw new ConfigException(entry.getWhere() + ": unknown setting '" + entry.getKey() + "'"
                    + Suggestions.hint(entry.getKey(), SETTERS.keySet()) + ". The command 'template' lists all settings");
            }
            setter.set(config, entry);
        }

        if(config.dataFile == null){
            throw new ConfigException(file.getSource() + ": data.file is required, it is the CSV file to train on");
        }
        if(config.dropout > 0.0 && config.hidden.isEmpty()){
            throw new ConfigException(file.getSource() + ": model.dropout needs at least one hidden layer, but model.hidden is empty");
        }
        for(String name : config.categorical){
            if(config.numeric.contains(name)){
                throw new ConfigException(file.getSource() + ": the column '" + name + "' is in both data.categorical and data.numeric");
            }
        }
        return config;
    }

    private static ConfigException fail(ConfigFile.Entry e, String problem){
        return new ConfigException(e.getWhere() + ": " + e.getKey() + " " + problem + ", got '" + e.getValue() + "'");
    }

    private static String asText(ConfigFile.Entry e){
        if(e.getValue().isEmpty()){
            throw new ConfigException(e.getWhere() + ": " + e.getKey() + " must not be empty");
        }
        return e.getValue();
    }

    private static int asInt(ConfigFile.Entry e, int min, int max){
        try{
            int value = Integer.parseInt(e.getValue());
            if(value < min || value > max){
                throw fail(e, "must be between " + min + " and " + max);
            }
            return value;
        }catch(NumberFormatException ex){
            throw fail(e, "must be a whole number");
        }
    }

    private static long asLong(ConfigFile.Entry e){
        try{
            return Long.parseLong(e.getValue());
        }catch(NumberFormatException ex){
            throw fail(e, "must be a whole number");
        }
    }

    private static double asDouble(ConfigFile.Entry e, double min, double max){
        try{
            double value = Double.parseDouble(e.getValue());
            if(Double.isNaN(value) || value < min || value > max){
                throw fail(e, "must be a number between " + min + " and " + max);
            }
            return value;
        }catch(NumberFormatException ex){
            throw fail(e, "must be a number");
        }
    }

    private static boolean asBool(ConfigFile.Entry e){
        String text = e.getValue().toLowerCase(Locale.ROOT);
        if(text.equals("true") || text.equals("yes") || text.equals("on") || text.equals("1")){
            return true;
        }
        if(text.equals("false") || text.equals("no") || text.equals("off") || text.equals("0")){
            return false;
        }
        throw fail(e, "must be true or false");
    }

    private static String asChoice(ConfigFile.Entry e, String[] choices){
        String text = e.getValue().toLowerCase(Locale.ROOT);
        for(String choice : choices){
            if(choice.equals(text)){
                return choice;
            }
        }
        throw fail(e, "must be one of " + Arrays.toString(choices) + Suggestions.hint(text, Arrays.asList(choices)));
    }

    private static List<String> asList(ConfigFile.Entry e){
        List<String> items = new ArrayList<>();
        for(String part : e.getValue().split(",")){
            String item = part.trim();
            if(!item.isEmpty()){
                items.add(item);
            }
        }
        return items;
    }

    private static List<Integer> asIntList(ConfigFile.Entry e, int min, int max, int maxCount){
        List<Integer> numbers = new ArrayList<>();

        if(e.getValue().equalsIgnoreCase("none")){
            return numbers;
        }
        for(String item : asList(e)){
            try{
                int value = Integer.parseInt(item);
                if(value < min || value > max){
                    throw fail(e, "has the size " + value + ", every size must be between " + min + " and " + max);
                }
                numbers.add(value);
            }catch(NumberFormatException ex){
                throw fail(e, "must be a list of whole numbers like 32, 16 (or none)");
            }
        }
        if(numbers.size() > maxCount){
            throw fail(e, "has " + numbers.size() + " layers, at most " + maxCount + " are allowed");
        }
        return numbers;
    }

    public DataOptions toDataOptions(){
        DataOptions options = new DataOptions()
            .testFraction(testFraction).seed(dataSeed).imputation(impute).missingIndicators(missingIndicators).scale(scale)
            .ignoreCase(ignoreCase).maxCategories(maxCategories).maxClasses(maxClasses).maxMissingFraction(maxMissingFraction)
            .badNumbersAsMissing(badNumbersAsMissing).forceClassification(forceClassification)
            .categorical(categorical.toArray(new String[0])).numeric(numeric.toArray(new String[0]));
        for(String token : missingTokens){
            options.missingToken(token);
        }
        return options;
    }

    public double effectiveLearningRate(){
        return Double.isNaN(learningRate) ? OptimizerFactory.defaultRate(optimizer) : learningRate;
    }

    public String describe(){
        StringBuilder sb = new StringBuilder();
        sb.append("data:     ").append(dataFile).append(", target ").append(target == null ? "(last column)" : target)
            .append(", test fraction ").append(testFraction).append("\n");
        sb.append("model:    hidden ").append(hidden.isEmpty() ? "none" : hidden.toString()).append(", ").append(activation)
            .append(", dropout ").append(dropout).append(", l2 ").append(l2).append("\n");
        sb.append("training: ").append(epochs).append(" epochs, batch ").append(batchSize).append(", ").append(optimizer)
            .append(" at ").append(effectiveLearningRate()).append(" (").append(schedule).append(")");
        if(patience > 0){
            sb.append(", early stopping after ").append(patience).append(" epochs on ").append(validationFraction).append(" of the training rows");
        }
        sb.append("\n");
        sb.append("output:   ").append(modelPath).append("\n");
        return sb.toString();
    }

    public static String template(){
        return String.join("\n",
            "# NeuralForge run settings. Lines starting with # are comments.",
            "# Every setting except data.file is optional, the values shown are the defaults.",
            "# Any setting can be changed from the command line: train this.cfg --set train.epochs=50",
            "",
            "[data]",
            "file = data/your_file.csv           # the CSV file, it needs a header row. Required",
            "# target = label                    # the column to predict, by name. Default: the last column",
            "test_fraction = 0.2                 # share of the rows kept for testing",
            "seed = 42                           # makes the split reproducible",
            "# missing_tokens = none, -          # more texts that mean 'missing' (empty, ?, NA, N/A, NaN, null always do)",
            "# categorical = zip_code            # columns to treat as categories even if they hold numbers",
            "# numeric = rating                  # columns to treat as numbers even if they look like categories",
            "impute = median                     # how holes in numbers are filled: median or mean",
            "missing_indicators = false          # add a 0/1 column telling which numbers were missing",
            "scale = true                        # scale numbers to mean 0 and standard deviation 1",
            "ignore_case = true                  # Red and red are one category",
            "max_categories = 20                 # a text column with more different values is dropped",
            "max_classes = 20                    # the target may have at most this many classes",
            "max_missing_fraction = 0.5          # a column with more holes than this is dropped",
            "bad_numbers_as_missing = false      # a stray text in a number column: false = stop, true = treat as missing",
            "force_classification = false        # accept a numeric target as class codes",
            "",
            "[model]",
            "hidden = 16                         # sizes of the hidden layers, like 32, 16. 'none' gives no hidden layer",
            "activation = relu                   # relu, leaky_relu, sigmoid or tanh",
            "init = auto                         # auto (he for relu, xavier otherwise), he or xavier",
            "dropout = 0.0                       # share of hidden neurons switched off while training",
            "l2 = 0.0                            # strength of the L2 penalty",
            "seed = 2026                         # starting weights and dropout",
            "",
            "[train]",
            "epochs = 100",
            "batch_size = 32",
            "optimizer = adam                    # " + String.join(", ", OptimizerFactory.names()),
            "# learning_rate = 0.001             # default: the usual rate of the optimizer",
            "schedule = constant                 # " + String.join(", ", ScheduleFactory.names()),
            "patience = 0                        # 0 = no early stopping, otherwise stop after this many epochs without improvement",
            "validation_fraction = 0.15          # share of the training rows watched by early stopping",
            "seed = 2026                         # order of the rows in every epoch",
            "print_every = 10                    # print a line every this many epochs, 0 = silent",
            "",
            "[output]",
            "model = models/model.nfm            # the preprocessor is saved next to it as models/model.nfp",
            "# history = results/history.csv     # loss and accuracy of every epoch",
            "");
    }

    public String getDataFile(){
        return dataFile;
    }

    public String getTarget(){
        return target;
    }

    public List<Integer> getHidden(){
        return new ArrayList<>(hidden);
    }

    public String getActivation(){
        return activation;
    }

    public String getInit(){
        return init;
    }

    public double getDropout(){
        return dropout;
    }

    public double getL2(){
        return l2;
    }

    public long getModelSeed(){
        return modelSeed;
    }

    public int getEpochs(){
        return epochs;
    }

    public int getBatchSize(){
        return batchSize;
    }

    public String getOptimizer(){
        return optimizer;
    }

    public String getSchedule(){
        return schedule;
    }

    public int getPatience(){
        return patience;
    }

    public double getValidationFraction(){
        return validationFraction;
    }

    public long getTrainSeed(){
        return trainSeed;
    }

    public int getPrintEvery(){
        return printEvery;
    }

    public String getModelPath(){
        return modelPath;
    }

    public String getHistoryPath(){
        return historyPath;
    }

    public double getTestFraction(){
        return testFraction;
    }

    public long getDataSeed(){
        return dataSeed;
    }

    public Imputer.Strategy getImpute(){
        return impute;
    }

    public boolean hasMissingIndicators(){
        return missingIndicators;
    }

    public boolean isScaling(){
        return scale;
    }

    public List<String> getMissingTokens(){
        return new ArrayList<>(missingTokens);
    }

    public List<String> getCategorical(){
        return new ArrayList<>(categorical);
    }

    public List<String> getNumeric(){
        return new ArrayList<>(numeric);
    }

    public int getMaxCategories(){
        return maxCategories;
    }

    public boolean isForcingClassification(){
        return forceClassification;
    }

    public boolean isBadNumbersAsMissing(){
        return badNumbersAsMissing;
    }
}
