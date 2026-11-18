package neuralforge.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class CsvReader{

    private static final char[] CANDIDATES = {',', ';', '\t', '|'};

    private static final int SAMPLE_LINES = 20;

    private CsvReader(){
    }

    public static CsvTable read(String path, boolean hasHeader){
        return read(path, hasHeader, ',');
    }

    public static CsvTable readAuto(String path, boolean hasHeader){
        return read(path, hasHeader, detectDelimiter(path));
    }

    public static CsvTable read(String path, boolean hasHeader, char delimiter){
        List<String[]> rows = new ArrayList<>();
        List<Integer> lineNumbers = new ArrayList<>();
        String[] header = null;

        int expected = -1;

        try(BufferedReader reader = open(path)){
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
                    throw new DataFormatException("line " + lineNumber + " has " + fields.length + " fields but the first line has " + expected
                        + ". Look for a missing or extra delimiter, or a text value that contains the delimiter but is not in quotes");
                }

                rows.add(fields);
                lineNumbers.add(lineNumber);
            }
        }catch(CharacterCodingException e){

            throw new DataFormatException(path + " is not a plain text file (it is not valid UTF-8). "
                + "If it is an Excel file, save it as CSV UTF-8 first");
        }catch(IOException e){
            throw new UncheckedIOException("could not read " + path + ": " + e.getMessage(), e);
        }

        if(expected == -1){
            throw new DataFormatException(path + " is empty");
        }

        if(header == null){
            header = new String[expected];
            for(int i = 0; i < expected; i++){
                header[i] = "col" + i;
            }
        }

        return new CsvTable(header, rows, lineNumbers);
    }

    public static char detectDelimiter(String path){
        List<String> sample = new ArrayList<>();
        try(BufferedReader reader = open(path)){
            String line;
            while(sample.size() < SAMPLE_LINES && (line = reader.readLine()) != null){
                if(sample.isEmpty() && line.startsWith("\uFEFF")){
                    line = line.substring(1);
                }
                if(!line.trim().isEmpty()){
                    sample.add(line);
                }
            }
        }catch(CharacterCodingException e){
            throw new DataFormatException(path + " is not a plain text file (it is not valid UTF-8). "
                + "If it is an Excel file, save it as CSV UTF-8 first");
        }catch(IOException e){
            throw new UncheckedIOException("could not read " + path + ": " + e.getMessage(), e);
        }

        if(sample.isEmpty()){
            return ',';
        }

        char best = ',';
        int bestCount = 0;
        for(char candidate : CANDIDATES){
            int first = countOutsideQuotes(sample.get(0), candidate);
            if(first == 0){
                continue;
            }
            boolean consistent = true;
            for(String line : sample){
                if(countOutsideQuotes(line, candidate) != first){
                    consistent = false;
                    break;
                }
            }
            if(consistent && first > bestCount){
                best = candidate;
                bestCount = first;
            }
        }
        return best;
    }

    private static BufferedReader open(String path) throws IOException{
        Path file = Paths.get(path);
        if(Files.isDirectory(file)){
            throw new DataFormatException(path + " is a folder, not a file");
        }
        if(!Files.exists(file)){
            throw new DataFormatException("file not found: " + path + " (the program looked in " + file.toAbsolutePath().getParent() + ")");
        }
        return Files.newBufferedReader(file, StandardCharsets.UTF_8);
    }

    private static int countOutsideQuotes(String line, char target){
        int count = 0;
        boolean inQuotes = false;
        for(int i = 0; i < line.length(); i++){
            char c = line.charAt(i);
            if(c == '"'){
                inQuotes = !inQuotes;
            }else if(c == target && !inQuotes){
                count++;
            }
        }
        return count;
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
                    throw new DataFormatException("line " + lineNumber + ": unexpected '" + c + "' after a closing quote");
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
            throw new DataFormatException("line " + lineNumber + ": a quote is opened but never closed");
        }

        fields.add(wasQuoted ? current.toString() : current.toString().trim());

        return fields.toArray(new String[0]);
    }
}
