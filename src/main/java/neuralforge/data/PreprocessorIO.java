package neuralforge.data;

import neuralforge.preprocess.Imputer;
import neuralforge.preprocess.OneHotEncoder;
import neuralforge.preprocess.StandardScaler;
import java.util.ArrayList;
import java.util.List;

public final class PreprocessorIO{

    private static final int MAX_COLUMNS = 100000;
    private static final int MAX_CATEGORIES = 1000000;

    private PreprocessorIO(){
    }

    public static void write(TablePreprocessor p, List<String> lines){
        DataOptions options = p.getOptions();
        lines.add(RecordText.line("ignore_case", String.valueOf(options.isIgnoringCase())));
        lines.add(RecordText.line("bad_numbers_as_missing", String.valueOf(options.isBadNumbersAsMissing())));
        lines.add(RecordText.line("imputation", options.getImputation().name()));

        List<String> tokens = options.getExtraMissingTokens();
        lines.add(RecordText.line("missing_tokens", String.valueOf(tokens.size())));
        for(String token : tokens){
            lines.add(RecordText.line("token", token));
        }

        List<String> dropped = p.getDroppedColumns();
        lines.add(RecordText.line("dropped", String.valueOf(dropped.size())));
        for(String text : dropped){
            lines.add(RecordText.line("drop", text));
        }

        List<String> numeric = p.getNumericColumns();
        lines.add(RecordText.line("scaled", String.valueOf(p.getScaler() != null)));
        lines.add(RecordText.line("numeric", String.valueOf(numeric.size())));
        for(int j = 0; j < numeric.size(); j++){
            double mean = p.getScaler() == null ? 0.0 : p.getScaler().getMean()[j];
            double std = p.getScaler() == null ? 1.0 : p.getScaler().getStd()[j];
            lines.add(RecordText.line("num", numeric.get(j), p.getIndicators()[j] ? "1" : "0",
                Double.toString(p.getImputer().getFill()[j]), Double.toString(mean), Double.toString(std)));
        }

        List<String> categorical = p.getCategoricalColumns();
        List<OneHotEncoder> encoders = p.getEncoders();
        lines.add(RecordText.line("categorical", String.valueOf(categorical.size())));
        for(int k = 0; k < categorical.size(); k++){
            String[] categories = encoders.get(k).getCategories();
            lines.add(RecordText.line("cat", categorical.get(k), p.getCategoryHasMissing().get(k) ? "1" : "0", String.valueOf(categories.length)));
            for(String category : categories){
                lines.add(RecordText.line("value", category));
            }
        }
    }

    public static TablePreprocessor read(LineCursor in){
        DataOptions options = new DataOptions();
        options.ignoreCase(in.nextBoolean("ignore_case"));
        options.badNumbersAsMissing(in.nextBoolean("bad_numbers_as_missing"));
        String strategy = in.nextText("imputation");
        try{
            options.imputation(Imputer.Strategy.valueOf(strategy));
        }catch(IllegalArgumentException e){
            throw in.error("unknown imputation '" + strategy + "'");
        }

        int tokenCount = in.nextInt("missing_tokens", 0, MAX_COLUMNS);
        for(int i = 0; i < tokenCount; i++){
            options.missingToken(in.nextText("token"));
        }

        int droppedCount = in.nextInt("dropped", 0, MAX_COLUMNS);
        List<String> dropped = new ArrayList<>();
        for(int i = 0; i < droppedCount; i++){
            dropped.add(in.nextText("drop"));
        }

        boolean scaled = in.nextBoolean("scaled");
        options.scale(scaled);
        int numericCount = in.nextInt("numeric", 0, MAX_COLUMNS);
        List<String> numericNames = new ArrayList<>();
        boolean[] indicators = new boolean[numericCount];
        double[] fill = new double[numericCount];
        double[] mean = new double[numericCount];
        double[] std = new double[numericCount];
        for(int j = 0; j < numericCount; j++){
            String[] f = in.next("num", 5);
            numericNames.add(f[0]);
            indicators[j] = f[1].equals("1");
            fill[j] = in.parseDouble(f[2]);
            mean[j] = in.parseDouble(f[3]);
            std[j] = in.parseDouble(f[4]);
        }

        int categoricalCount = in.nextInt("categorical", 0, MAX_COLUMNS);
        List<String> categoricalNames = new ArrayList<>();
        List<OneHotEncoder> encoders = new ArrayList<>();
        List<Boolean> hasMissing = new ArrayList<>();
        for(int k = 0; k < categoricalCount; k++){
            String[] f = in.next("cat", 3);
            categoricalNames.add(f[0]);
            hasMissing.add(f[1].equals("1"));
            int size;
            try{
                size = Integer.parseInt(f[2]);
            }catch(NumberFormatException e){
                throw in.error("the number of categories of '" + f[0] + "' is not a number");
            }
            if(size < 1 || size > MAX_CATEGORIES){
                throw in.error("the column '" + f[0] + "' claims to have " + size + " categories");
            }
            String[] categories = new String[size];
            for(int c = 0; c < size; c++){
                categories[c] = in.nextText("value");
            }
            try{
                encoders.add(OneHotEncoder.restore(categories));
            }catch(IllegalArgumentException e){
                throw in.error(e.getMessage());
            }
        }

        try{
            Imputer imputer = numericCount > 0 ? Imputer.restore(options.getImputation(), fill) : null;
            StandardScaler scaler = scaled && numericCount > 0 ? StandardScaler.restore(mean, std) : null;
            return TablePreprocessor.restore(options, numericNames, indicators, imputer, scaler, categoricalNames, encoders, hasMissing, dropped);
        }catch(IllegalArgumentException e){
            throw new DataFormatException("the saved preprocessor cannot be used: " + e.getMessage());
        }
    }
}
