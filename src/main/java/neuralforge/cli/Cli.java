package neuralforge.cli;

import neuralforge.data.CsvReader;
import neuralforge.data.CsvTable;
import neuralforge.data.DataReport;
import neuralforge.data.TableProfiler;
import neuralforge.io.ModelBundle;
import neuralforge.metrics.ConfusionMatrix;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class Cli{

    public static final int OK = 0;
    public static final int PROBLEM = 1;
    public static final int USAGE = 2;
    public static final int BUG = 3;

    private static final String[] COMMANDS = {"train", "predict", "evaluate", "inspect", "template", "help"};

    private Cli(){
    }

    public static boolean isCommand(String word){
        return Arrays.asList(COMMANDS).contains(word.toLowerCase(Locale.ROOT)) || word.equals("-h") || word.equals("--help");
    }

    public static List<String> commandNames(){
        return new ArrayList<>(Arrays.asList(COMMANDS));
    }

    public static int run(String[] args, PrintStream out, PrintStream err){
        try{
            if(args.length == 0){
                out.print(overview());
                return OK;
            }
            String command = args[0].toLowerCase(Locale.ROOT);
            switch(command){
                case "train":
                    return train(args, out);
                case "predict":
                    return predict(args, out);
                case "evaluate":
                    return evaluate(args, out);
                case "inspect":
                    return inspect(args, out);
                case "template":
                    return template(args, out);
                case "help":
                case "-h":
                case "--help":
                    return help(args, out);
                default:
                    throw new UsageException("unknown command '" + args[0] + "'" + Suggestions.hint(command, commandNames()));
            }
        }catch(UsageException e){
            err.println("error: " + e.getMessage());
            err.println("run 'help' to see the commands");
            return USAGE;
        }catch(IllegalArgumentException e){

            err.println("error: " + e.getMessage());
            return PROBLEM;
        }catch(UncheckedIOException e){
            err.println("error: " + e.getMessage());
            return PROBLEM;
        }catch(RuntimeException e){
            err.println("internal error, this is a bug in the program: " + e);
            return BUG;
        }
    }

    private static int train(String[] args, PrintStream out){
        ArgParser.Parsed parsed = new ArgParser(new String[]{"set"}, new String[0], new String[]{"set"}).parse(args, 1);
        requirePositionals(parsed, 1, "train <config file> [--set key=value]...");

        RunConfig config = RunConfig.fromFile(parsed.positionals().get(0), parsed.values("set"));
        TrainingRun.run(config, out);
        return OK;
    }

    private static int predict(String[] args, PrintStream out){
        ArgParser.Parsed parsed = new ArgParser(new String[]{"output"}, new String[]{"no-probabilities"}, new String[0]).parse(args, 1);
        requirePositionals(parsed, 2, "predict <model> <data.csv> [--output file.csv] [--no-probabilities]");

        String modelPath = parsed.positionals().get(0);
        String dataPath = parsed.positionals().get(1);
        String outputPath = parsed.value("output") != null ? parsed.value("output") : defaultOutput(dataPath);

        ModelBundle bundle = ModelBundle.load(modelPath);
        char delimiter = CsvReader.detectDelimiter(dataPath);
        CsvTable table = CsvReader.read(dataPath, true, delimiter);
        if(table.rowCount() == 0){
            throw new IllegalArgumentException(dataPath + " has a header but no rows");
        }

        DataReport report = new DataReport();
        Predictor predictor = new Predictor(bundle);
        PredictionResult result = predictor.predict(table, null, report);

        writeFile(outputPath, writer -> predictor.writeCsv(table, TableProfiler.allRows(table.rowCount()), result, !parsed.has("no-probabilities"), delimiter, writer));

        out.println("predicted " + table.rowCount() + " rows from " + dataPath);
        String[] names = result.getClassNames();
        int[] counts = result.classCounts();
        for(int c = 0; c < names.length; c++){
            out.println(String.format(Locale.US, "  %-20s %6d  (%.1f%%)", names[c], counts[c], 100.0 * counts[c] / result.size()));
        }
        printWarnings(report, out);
        out.println("answers written to " + outputPath);
        return OK;
    }

    private static int evaluate(String[] args, PrintStream out){
        ArgParser.Parsed parsed = new ArgParser(new String[]{"errors"}, new String[0], new String[0]).parse(args, 1);
        requirePositionals(parsed, 2, "evaluate <model> <data.csv> [--errors file.csv]");

        ModelBundle bundle = ModelBundle.load(parsed.positionals().get(0));
        String dataPath = parsed.positionals().get(1);
        char delimiter = CsvReader.detectDelimiter(dataPath);
        CsvTable table = CsvReader.read(dataPath, true, delimiter);

        DataReport report = new DataReport();
        Evaluator.Result result = Evaluator.evaluate(bundle, table, report);

        out.println("evaluated " + result.getUsedCount() + " of " + result.getRowCount() + " rows from " + dataPath);
        printWarnings(report, out);
        out.println();

        String[] names = bundle.getClassNames();
        out.print(result.getMatrix().format(names));
        out.print(result.getMatrix().report(names));
        out.println(String.format(Locale.US, "accuracy %.2f%%, loss %.4f, %d wrong", 100 * result.getAccuracy(), result.getLoss(), result.wrongCount()));

        ConfusionMatrix matrix = result.getMatrix();
        int biggest = 0;
        for(int c = 0; c < names.length; c++){
            biggest = Math.max(biggest, matrix.support(c));
        }
        out.println(String.format(Locale.US, "always answering the most common class would give %.2f%%", 100.0 * biggest / matrix.total()));

        String errorsPath = parsed.value("errors");
        if(errorsPath != null){
            if(result.wrongCount() == 0){
                out.println("no mistakes, so " + errorsPath + " was not written");
            }else{
                writeFile(errorsPath, writer -> new Predictor(bundle).writeCsv(table, result.wrongTableRows(), result.wrongPredictions(), true, delimiter, writer));
                out.println("the " + result.wrongCount() + " wrong rows with the answers were written to " + errorsPath);
            }
        }
        return OK;
    }

    private static int inspect(String[] args, PrintStream out){
        ArgParser.Parsed parsed = new ArgParser(new String[0], new String[0], new String[0]).parse(args, 1);
        requirePositionals(parsed, 1, "inspect <model>");

        String modelPath = parsed.positionals().get(0);
        ModelBundle bundle = ModelBundle.load(modelPath);

        out.println("model:    " + modelPath + " (preprocessor: " + ModelBundle.preprocessorPath(modelPath) + ")");
        out.println("task:     " + bundle.getTask().label() + ", target '" + bundle.getTargetName() + "'");
        out.println("classes:  " + String.join(", ", bundle.getClassNames()));
        String[] features = bundle.getPreprocessor().getFeatureNames();
        out.println("features: " + features.length + ": " + String.join(", ", features));
        List<String> dropped = bundle.getPreprocessor().getDroppedColumns();
        if(!dropped.isEmpty()){
            out.println("columns that were not used:");
            for(String text : dropped){
                out.println("  " + text);
            }
        }
        out.println();
        out.print(bundle.getNetwork().summary());
        Map<String, String> notes = bundle.getNotes();
        if(!notes.isEmpty()){
            out.println();
            out.println("training notes:");
            for(Map.Entry<String, String> note : notes.entrySet()){
                out.println("  " + note.getKey() + ": " + note.getValue());
            }
        }
        return OK;
    }

    private static int template(String[] args, PrintStream out){
        ArgParser.Parsed parsed = new ArgParser(new String[]{"output"}, new String[]{"force"}, new String[0]).parse(args, 1);
        requirePositionals(parsed, 0, "template [--output file.cfg] [--force]");

        String path = parsed.value("output");
        if(path == null){
            out.print(RunConfig.template());
            return OK;
        }
        if(Files.exists(Paths.get(path)) && !parsed.has("force")){
            throw new IllegalArgumentException(path + " already exists, use --force to replace it");
        }
        writeFile(path, writer -> writer.write(RunConfig.template()));
        out.println("example config written to " + path);
        return OK;
    }

    private static int help(String[] args, PrintStream out){
        if(args.length < 2){
            out.print(overview());
            return OK;
        }
        String command = args[1].toLowerCase(Locale.ROOT);
        switch(command){
            case "train":
                out.println("train <config file> [--set key=value]...");
                out.println();
                out.println("Prepares the data, trains the network and tests it, all as the config file says, and saves the");
                out.println("model (.nfm) and its preprocessor (.nfp). --set changes a setting without editing the file and can");
                out.println("be given many times, for example: train run.cfg --set train.epochs=50 --set model.dropout=0.2");
                out.println("The command 'template' prints a config file with all settings.");
                return OK;
            case "predict":
                out.println("predict <model> <data.csv> [--output file.csv] [--no-probabilities]");
                out.println();
                out.println("Answers for every row of the file. The file needs the same feature columns as the training file,");
                out.println("by name and in any order, and it does not need the target column. The answers are written as new");
                out.println("columns after the original ones: predicted_<target>, confidence and prob_<class> for every class.");
                out.println("Without --output the file is written next to the input, with _predictions added to its name.");
                return OK;
            case "evaluate":
                out.println("evaluate <model> <data.csv> [--errors file.csv]");
                out.println();
                out.println("Measures the model on a file that has the true labels in the target column: accuracy, loss, the");
                out.println("confusion matrix and precision, recall and F1 of every class. Rows without a label, or with a label");
                out.println("the model has never seen, are left out and counted. --errors writes the wrong rows to a file.");
                return OK;
            case "inspect":
                out.println("inspect <model>");
                out.println();
                out.println("Shows the task, the classes, the features, the network and the notes saved by the training run.");
                return OK;
            case "template":
                out.println("template [--output file.cfg] [--force]");
                out.println();
                out.println("Prints an example config file that lists every setting with its default and what it does.");
                out.println("With --output it is written to a file (an existing file is only replaced with --force).");
                return OK;
            case "help":
                out.println("help [command]");
                return OK;
            default:
                throw new UsageException("there is no command '" + args[1] + "'" + Suggestions.hint(command, commandNames()));
        }
    }

    private static String overview(){
        return String.join("\n",
            "NeuralForge command line",
            "",
            "  train     <config file>         train a model as the config file says, and save it",
            "  predict   <model> <data.csv>    answer for every row of a file, write the answers to a CSV",
            "  evaluate  <model> <data.csv>    measure a model on a file that has the true labels",
            "  inspect   <model>               show what a saved model is",
            "  template                        print an example config file with every setting",
            "  help      [command]             the details of one command",
            "",
            "A quick start:",
            "  template --output run.cfg       write an example config, then edit data.file in it",
            "  train run.cfg                   train",
            "  predict models/model.nfm new_rows.csv",
            "",
            "Exit codes: 0 ok, 1 problem with data, model or config, 2 wrong use of a command, 3 bug",
            "");
    }

    private static void requirePositionals(ArgParser.Parsed parsed, int count, String usage){
        if(parsed.positionals().size() < count){
            throw new UsageException("missing argument. Usage: " + usage);
        }
        if(parsed.positionals().size() > count){
            throw new UsageException("too many arguments, '" + parsed.positionals().get(count) + "' is not expected. Usage: " + usage);
        }
    }

    private static void printWarnings(DataReport report, PrintStream out){
        for(String message : report.getMessages()){
            out.println("  " + message);
        }
    }

    private static String defaultOutput(String dataPath){
        int slash = Math.max(dataPath.lastIndexOf('/'), dataPath.lastIndexOf('\\'));
        int dot = dataPath.lastIndexOf('.');
        String stem = dot > slash ? dataPath.substring(0, dot) : dataPath;
        return stem + "_predictions.csv";
    }

    private interface WriterAction{
        void write(Writer writer) throws IOException;
    }

    private static void writeFile(String path, WriterAction action){
        Path target = Paths.get(path);
        Path temp = Paths.get(path + ".tmp");
        try{
            Path folder = target.toAbsolutePath().getParent();
            if(folder != null){
                Files.createDirectories(folder);
            }
            try(Writer writer = new BufferedWriter(new OutputStreamWriter(Files.newOutputStream(temp), StandardCharsets.UTF_8))){
                action.write(writer);
            }
            Files.move(temp, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }catch(IOException e){
            try{
                Files.deleteIfExists(temp);
            }catch(IOException ignored){
            }
            throw new UncheckedIOException("could not write " + path + ": " + e.getMessage(), e);
        }
    }
}
