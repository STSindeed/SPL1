package neuralforge.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CsvTable{

    private final String[] header;
    private final List<String[]> rows;

    private final List<Integer> lineNumbers;

    public CsvTable(String[] header, List<String[]> rows, List<Integer> lineNumbers){
        if(header.length == 0){
            throw new IllegalArgumentException("a table needs at least one column");
        }
        if(rows.size() != lineNumbers.size()){
            throw new IllegalArgumentException("every row needs a line number");
        }
        for(int i = 0; i < rows.size(); i++){
            if(rows.get(i).length != header.length){
                throw new IllegalArgumentException("row " + i + " has " + rows.get(i).length + " fields but the header has " + header.length);
            }
        }
        this.header = header.clone();
        this.rows = new ArrayList<>(rows);
        this.lineNumbers = new ArrayList<>(lineNumbers);
    }

    public int rowCount(){
        return rows.size();
    }

    public int columnCount(){
        return header.length;
    }

    public String[] getHeader(){
        return header.clone();
    }

    public String get(int row, int column){
        return rows.get(row)[column];
    }

    public int lineNumber(int row){
        return lineNumbers.get(row);
    }

    public int columnIndex(String name){
        for(int i = 0; i < header.length; i++){
            if(header[i].equals(name)){
                return i;
            }
        }
        throw new IllegalArgumentException("no column named '" + name + "', the columns are " + Arrays.toString(header));
    }
}
