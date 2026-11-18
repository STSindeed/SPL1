package neuralforge.data;

import java.util.Locale;

public final class TaskDetector{

    public static class Detection{

        private final TaskType task;
        private final String reason;

        Detection(TaskType task, String reason){
            this.task = task;
            this.reason = reason;
        }

        public TaskType getTask(){
            return task;
        }

        public String getReason(){
            return reason;
        }
    }

    private TaskDetector(){
    }

    public static Detection detect(ColumnProfile target, int maxClasses){
        String name = target.getName();
        int distinct = target.getDistinctCount();

        if(target.getKnownCount() == 0){
            throw new DataFormatException("the target column '" + name + "' has no values at all");
        }
        if(distinct == 1){
            throw new DataFormatException("the target column '" + name + "' has the same value ('" + target.getTopValue()
                + "') in every row, there is nothing to learn from it");
        }
        if(distinct == 2){
            return new Detection(TaskType.BINARY_CLASSIFICATION, "the target has exactly two distinct values");
        }

        if(target.getType() == ColumnType.CATEGORICAL){
            if(distinct > maxClasses){
                throw new DataFormatException("the target column '" + name + "' is text with " + distinct + " distinct values, more than "
                    + maxClasses + ". That looks like free text or an identifier, not class labels. Check that the right column is "
                    + "the target, or raise maxClasses if there really are that many classes");
            }
            return new Detection(TaskType.MULTICLASS_CLASSIFICATION, "the target is text with " + distinct + " distinct values");
        }

        if(target.isAllWhole() && distinct <= maxClasses){
            return new Detection(TaskType.MULTICLASS_CLASSIFICATION,
                "the target holds whole numbers with only " + distinct + " distinct values, so they are read as class codes");
        }
        String why = target.isAllWhole()
            ? "the target holds whole numbers but has " + distinct + " distinct values (more than " + maxClasses + ")"
            : "the target holds numbers with decimals and " + distinct + " distinct values";
        return new Detection(TaskType.REGRESSION, why + String.format(Locale.US, ", ranging from %.4g to %.4g", target.getMin(), target.getMax()));
    }
}
