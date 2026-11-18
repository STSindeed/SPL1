package neuralforge.data;

public class DataFormatException extends IllegalArgumentException{

    public DataFormatException(String message){
        super(message);
    }

    public static DataFormatException at(int lineNumber, String column, String message){
        return new DataFormatException("line " + lineNumber + ", column '" + column + "': " + message);
    }
}
