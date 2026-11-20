package neuralforge.io;

import neuralforge.data.DataFormatException;
import neuralforge.data.DataPipeline;
import neuralforge.data.LineCursor;
import neuralforge.data.PreparedData;
import neuralforge.data.PreprocessorIO;
import neuralforge.data.RecordText;
import neuralforge.data.TablePreprocessor;
import neuralforge.data.TaskType;
import neuralforge.network.NeuralNetwork;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.CRC32;

public class ModelBundle{

    private static final String HEADER = "NeuralForge bundle";
    private static final int FORMAT_VERSION = 1;
    private static final int MAX_CLASSES = 10000;
    private static final int MAX_NOTES = 1000;

    private final NeuralNetwork network;
    private final TablePreprocessor preprocessor;
    private final String[] classNames;
    private final String targetName;
    private final boolean numericTarget;
    private final TaskType task;

    private final Map<String, String> notes;
    private final Map<String, Integer> classIndex = new HashMap<>();

    public ModelBundle(NeuralNetwork network, TablePreprocessor preprocessor, String[] classNames, String targetName,
                       boolean numericTarget, TaskType task, Map<String, String> notes){
        if(network.getLayers().isEmpty()){
            throw new IllegalArgumentException("the network has no layers");
        }
        if(network.getLayers().get(0).inputSize() != preprocessor.featureCount()){
            throw new DataFormatException("the model takes " + network.getLayers().get(0).inputSize() + " inputs but the preprocessor produces "
                + preprocessor.featureCount() + " features, they do not belong together");
        }
        int outputs = network.getLayers().get(network.getLayers().size() - 1).outputSize();
        if(outputs != classNames.length){
            throw new DataFormatException("the model has " + outputs + " outputs but there are " + classNames.length + " class names");
        }
        this.network = network;
        this.preprocessor = preprocessor;
        this.classNames = classNames.clone();
        this.targetName = targetName;
        this.numericTarget = numericTarget;
        this.task = task;
        this.notes = new LinkedHashMap<>(notes);
        for(int c = 0; c < classNames.length; c++){
            classIndex.put(classNames[c], c);
        }
    }

    public static ModelBundle of(NeuralNetwork network, PreparedData data, Map<String, String> notes){
        return new ModelBundle(network, data.getPreprocessor(), data.getClassNames(), data.getTargetName(), data.isNumericTarget(), data.getTask(), notes);
    }

    public static String preprocessorPath(String modelPath){
        int slash = Math.max(modelPath.lastIndexOf('/'), modelPath.lastIndexOf('\\'));
        int dot = modelPath.lastIndexOf('.');
        String stem = dot > slash ? modelPath.substring(0, dot) : modelPath;
        return stem + ".nfp";
    }

    public void save(String modelPath){

        ModelIO.save(network, modelPath);

        List<String> lines = new ArrayList<>();
        lines.add(RecordText.line(HEADER, String.valueOf(FORMAT_VERSION)));
        lines.add(RecordText.line("target", targetName));
        lines.add(RecordText.line("target_numeric", String.valueOf(numericTarget)));
        lines.add(RecordText.line("task", task.name()));
        lines.add(RecordText.line("classes", String.valueOf(classNames.length)));
        for(String name : classNames){
            lines.add(RecordText.line("class", name));
        }
        lines.add(RecordText.line("notes", String.valueOf(notes.size())));
        for(Map.Entry<String, String> note : notes.entrySet()){
            lines.add(RecordText.line("note", note.getKey(), note.getValue()));
        }
        PreprocessorIO.write(preprocessor, lines);

        lines.add(RecordText.line("crc", Long.toString(checksum(lines))));

        String path = preprocessorPath(modelPath);
        Path target = Paths.get(path);
        Path temp = Paths.get(path + ".tmp");
        try{
            Files.write(temp, (String.join("\n", lines) + "\n").getBytes(StandardCharsets.UTF_8));
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING);
        }catch(IOException e){
            try{
                Files.deleteIfExists(temp);
            }catch(IOException ignored){
            }
            throw new UncheckedIOException("could not write " + path + ": " + e.getMessage(), e);
        }
    }

    public static ModelBundle load(String modelPath){
        if(!Files.exists(Paths.get(modelPath))){
            throw new DataFormatException("model file not found: " + modelPath);
        }
        String textPath = preprocessorPath(modelPath);
        if(!Files.exists(Paths.get(textPath))){
            throw new DataFormatException("the model " + modelPath + " has no " + textPath + " next to it. That file holds the preprocessor, "
                + "and without it new data cannot be turned into the numbers the model expects. Train the model again with the train command");
        }

        List<String> lines;
        try{
            lines = Files.readAllLines(Paths.get(textPath), StandardCharsets.UTF_8);
        }catch(IOException e){
            throw new DataFormatException(textPath + " could not be read as text: " + e.getMessage());
        }

        if(lines.isEmpty()){
            throw new DataFormatException(textPath + " is empty");
        }
        String[] last = RecordText.fields(lines.get(lines.size() - 1));
        if(last.length != 2 || !last[0].equals("crc")){
            throw new DataFormatException(textPath + " has no checksum at the end, it is damaged or incomplete");
        }
        List<String> body = new ArrayList<>(lines.subList(0, lines.size() - 1));
        if(!last[1].equals(Long.toString(checksum(body)))){
            throw new DataFormatException(textPath + " is damaged, its checksum does not match. It must not be edited by hand");
        }

        LineCursor in = new LineCursor(body, textPath);
        String[] header = in.next(HEADER, 1);
        if(!header[0].equals(String.valueOf(FORMAT_VERSION))){
            throw in.error("format version " + header[0] + ", this program reads version " + FORMAT_VERSION);
        }
        String target = in.nextText("target");
        boolean numeric = in.nextBoolean("target_numeric");
        String taskName = in.nextText("task");
        TaskType task;
        try{
            task = TaskType.valueOf(taskName);
        }catch(IllegalArgumentException e){
            throw in.error("unknown task '" + taskName + "'");
        }
        int classCount = in.nextInt("classes", 2, MAX_CLASSES);
        String[] classes = new String[classCount];
        for(int c = 0; c < classCount; c++){
            classes[c] = in.nextText("class");
        }
        int noteCount = in.nextInt("notes", 0, MAX_NOTES);
        Map<String, String> notes = new LinkedHashMap<>();
        for(int i = 0; i < noteCount; i++){
            String[] note = in.next("note", 2);
            notes.put(note[0], note[1]);
        }
        TablePreprocessor preprocessor = PreprocessorIO.read(in);
        if(in.hasNext()){
            throw in.error("unexpected extra lines after the preprocessor");
        }

        NeuralNetwork network = ModelIO.load(modelPath);
        return new ModelBundle(network, preprocessor, classes, target, numeric, task, notes);
    }

    public int classIndexOf(String labelText){
        Integer index = classIndex.get(DataPipeline.labelText(labelText, numericTarget, preprocessor.getOptions()));
        return index == null ? -1 : index;
    }

    public boolean isMissingLabel(String labelText){
        return preprocessor.getOptions().isMissing(labelText);
    }

    private static long checksum(List<String> lines){
        CRC32 crc = new CRC32();
        crc.update((String.join("\n", lines) + "\n").getBytes(StandardCharsets.UTF_8));
        return crc.getValue();
    }

    public NeuralNetwork getNetwork(){
        return network;
    }

    public TablePreprocessor getPreprocessor(){
        return preprocessor;
    }

    public String[] getClassNames(){
        return classNames.clone();
    }

    public String getTargetName(){
        return targetName;
    }

    public boolean isNumericTarget(){
        return numericTarget;
    }

    public TaskType getTask(){
        return task;
    }

    public Map<String, String> getNotes(){
        return new LinkedHashMap<>(notes);
    }
}
