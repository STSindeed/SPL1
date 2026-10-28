package neuralforge.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class CsvReader{

    private CsvReader(){
    }

    public static CsvTable read(String path, boolean hasHeader){
        return read(path, hasHeader, ',');
    }

    public static CsvTable read(String path, boolean hasHeader, char delimiter){
        List<String[]> rows = new ArrayList<>();
        List<Integer> lineNumbers = new ArrayList<>();
        String[] header = null;

        int expected = -1;

        try(BufferedReader reader = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)){
            String line;
            int lineNumber = 0;

            while((line = reader.readLine()) != null){
                lineNumber++;

                if(lineNumber == 1 && line.startsWith("\uFEFF")){
                    line = line.substring(1);
                }

                if(line.trim().isEmpty()){
                    continue;
                }

                String[] fields = parseLine(line, delimiter, lineNumber);

                if(expected == -1){
                    expected = fields.length;
                    if(hasHeader){
                        header = fields;
                        continue;
                    }
                }else if(fields.length != expected){
                    throw new IllegalArgumentException("line " + lineNumber + " has " + fields.length + " fields but the first line has " + expected);
                }

                rows.add(fields);
                lineNumbers.add(lineNumber);
            }
        }catch(IOException e){
            throw new UncheckedIOException("could not read " + path + ": " + e.getMessage(), e);
        }

        if(expected == -1){
            throw new IllegalArgumentException(path + " is empty");
        }

        if(header == null){
            header = new String[expected];
            for(int i = 0; i < expected; i++){
                header[i] = "col" + i;
            }
        }

        return new CsvTable(header, rows, lineNumbers);
    }

    static String[] parseLine(String line, char delimiter, int lineNumber){
        List<String> fields = new ArrayList<>();
        StringBuilder current = new StringBuilder();

        boolean inQuotes = false;

        boolean wasQuoted = false;

        boolean atFieldStart = true;

        int i = 0;
        while(i < line.length()){
            char c = line.charAt(i);

            if(inQuotes){
                if(c == '"'){
                    boolean doubled = i + 1 < line.length() && line.charAt(i + 1) == '"';
                    if(doubled){
                        current.append('"');
                        i++;
                    }else{
                        inQuotes = false;
                    }
                }else{
                    current.append(c);
                }
            }else if(c == delimiter){
                fields.add(wasQuoted ? current.toString() : current.toString().trim());
                current.setLength(0);
                wasQuoted = false;
                atFieldStart = true;
            }else if(wasQuoted){
                if(!Character.isWhitespace(c)){
                    throw new IllegalArgumentException("line " + lineNumber + ": unexpected '" + c + "' after a closing quote");
                }
            }else if(c == '"' && atFieldStart){
                current.setLength(0);
                inQuotes = true;
                wasQuoted = true;
                atFieldStart = false;
            }else{
                current.append(c);
                if(!Character.isWhitespace(c)){
                    atFieldStart = false;
                }
            }
            i++;
        }

        if(inQuotes){
            throw new IllegalArgumentException("line " + lineNumber + ": a quote is opened but never closed");
        }

        fields.add(wasQuoted ? current.toString() : current.toString().trim());

        return fields.toArray(new String[0]);
    }
}
