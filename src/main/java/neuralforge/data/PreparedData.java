package neuralforge.data;

import java.util.Locale;

public class PreparedData{

    private final Dataset train;
    private final Dataset test;
    private final TablePreprocessor preprocessor;
    private final TaskType task;
    private final String taskReason;

    private final boolean numericTarget;
    private final String targetName;
    private final ColumnProfile[] profiles;
    private final DataReport report;

    PreparedData(Dataset train, Dataset test, TablePreprocessor preprocessor, TaskType task, String taskReason,
                 boolean numericTarget, String targetName, ColumnProfile[] profiles, DataReport report){
        this.train = train;
        this.test = test;
        this.preprocessor = preprocessor;
        this.task = task;
        this.taskReason = taskReason;
        this.numericTarget = numericTarget;
        this.targetName = targetName;
        this.profiles = profiles;
        this.report = report;
    }

    public Dataset getTrain(){
        return train;
    }

    public Dataset getTest(){
        return test;
    }

    public TablePreprocessor getPreprocessor(){
        return preprocessor;
    }

    public TaskType getTask(){
        return task;
    }

    public String getTaskReason(){
        return taskReason;
    }

    public boolean isNumericTarget(){
        return numericTarget;
    }

    public String getTargetName(){
        return targetName;
    }

    public String[] getClassNames(){
        return train.getClassNames();
    }

    public ColumnProfile[] getProfiles(){
        return profiles.clone();
    }

    public DataReport getReport(){
        return report;
    }

    public String describe(){
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US, "task: %s (%s)\n", task.label(), taskReason));
        sb.append("target: ").append(targetName).append("\n");
        sb.append(train.size()).append(" training rows, ").append(test.size()).append(" test rows, ");
        sb.append(train.featureCount()).append(" features\n");
        sb.append("classes:");
        int[] counts = new int[train.classCount()];
        int[] trainCounts = train.classCounts();
        int[] testCounts = test.classCounts();
        for(int c = 0; c < counts.length; c++){
            counts[c] = trainCounts[c] + testCounts[c];
            sb.append(" ").append(train.className(c)).append(" (").append(counts[c]).append(")");
        }
        sb.append("\n");
        return sb.toString();
    }
}
