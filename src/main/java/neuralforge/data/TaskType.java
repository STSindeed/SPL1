package neuralforge.data;

public enum TaskType{

    BINARY_CLASSIFICATION("binary classification"),
    MULTICLASS_CLASSIFICATION("multiclass classification"),
    REGRESSION("regression");

    private final String label;

    TaskType(String label){
        this.label = label;
    }

    public String label(){
        return label;
    }

    public boolean isClassification(){
        return this != REGRESSION;
    }
}
