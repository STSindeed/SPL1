package neuralforge.cli;

import java.io.IOException;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ConfigFile{

    public static class Entry{

        private final String key;
        private final String value;
        private final String where;

        Entry(String key, String value, String where){
            this.key = key;
            this.value = value;
            this.where = where;
        }

        public String getKey(){
            return key;
        }

        public String getValue(){
            return value;
        }

        public String getWhere(){
            return where;
        }
    }

    private final Map<String, Entry> entries = new LinkedHashMap<>();
    private final String source;

    private ConfigFile(String source){
        this.source = source;
    }

    public static ConfigFile read(String path){
        Path file = Paths.get(path);
        if(Files.isDirectory(file)){
            throw new ConfigException(path + " is a folder, not a config file");
        }
        if(!Files.exists(file)){
            throw new ConfigException("config file not found: " + path);
        }
        try{
            return parse(String.join("\n", Files.readAllLines(file, StandardCharsets.UTF_8)), path);
        }catch(CharacterCodingException e){
            throw new ConfigException(path + " is not a plain text file (it is not valid UTF-8)");
        }catch(IOException e){
            throw new ConfigException("could not read " + path + ": " + e.getMessage());
        }
    }

    public static ConfigFile parse(String text, String source){
        ConfigFile config = new ConfigFile(source);
        String section = "";

        String[] lines = text.split("\n", -1);
        for(int i = 0; i < lines.length; i++){
            String where = source + ", line " + (i + 1);
            String line = lines[i];
            if(i == 0 && line.startsWith("\uFEFF")){
                line = line.substring(1);
            }
            line = line.trim();

            if(line.isEmpty() || line.startsWith("#") || line.startsWith(";")){
                continue;
            }

            if(line.startsWith("[")){
                int close = line.indexOf(']');
                if(close < 0){
                    throw new ConfigException(where + ": the section header '" + line + "' is missing its closing ]");
                }
                String rest = line.substring(close + 1).trim();
                if(!rest.isEmpty() && !rest.startsWith("#") && !rest.startsWith(";")){
                    throw new ConfigException(where + ": unexpected text '" + rest + "' after the section header");
                }
                section = line.substring(1, close).trim().toLowerCase();
                continue;
            }

            int equals = line.indexOf('=');
            if(equals < 0){
                throw new ConfigException(where + ": expected 'key = value' but found '" + line + "'");
            }
            String key = line.substring(0, equals).trim().toLowerCase();
            if(key.isEmpty()){
                throw new ConfigException(where + ": there is a value but no key before the =");
            }
            String value = cleanValue(line.substring(equals + 1), where);

            String fullKey = section.isEmpty() ? key : section + "." + key;
            Entry before = config.entries.get(fullKey);
            if(before != null){
                throw new ConfigException(where + ": '" + fullKey + "' is set twice, the first time was at " + before.getWhere());
            }
            config.entries.put(fullKey, new Entry(fullKey, value, where));
        }
        return config;
    }

    public void override(String assignment){
        int equals = assignment.indexOf('=');
        if(equals < 0){
            throw new UsageException("--set expects key=value, for example --set train.epochs=50, but got '" + assignment + "'");
        }
        String key = assignment.substring(0, equals).trim().toLowerCase();
        if(key.isEmpty()){
            throw new UsageException("--set has a value but no key: '" + assignment + "'");
        }
        String value = cleanValue(assignment.substring(equals + 1), "--set " + key);
        entries.put(key, new Entry(key, value, "--set " + key));
    }

    public List<Entry> entries(){
        return new ArrayList<>(entries.values());
    }

    public Entry get(String key){
        return entries.get(key);
    }

    public String getSource(){
        return source;
    }

    private static String cleanValue(String raw, String where){
        String value = raw.trim();

        if(value.startsWith("\"")){
            int close = value.indexOf('"', 1);
            if(close < 0){
                throw new ConfigException(where + ": a quote is opened but never closed");
            }
            String rest = value.substring(close + 1).trim();
            if(!rest.isEmpty() && !rest.startsWith("#") && !rest.startsWith(";")){
                throw new ConfigException(where + ": unexpected text '" + rest + "' after the closing quote");
            }
            return value.substring(1, close);
        }

        for(int i = 1; i < value.length(); i++){
            char c = value.charAt(i);
            if((c == '#' || c == ';') && Character.isWhitespace(value.charAt(i - 1))){
                return value.substring(0, i).trim();
            }
        }

        if(value.startsWith("#") || value.startsWith(";")){
            return "";
        }
        return value;
    }
}
