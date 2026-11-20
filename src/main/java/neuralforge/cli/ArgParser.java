package neuralforge.cli;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ArgParser{

    public static class Parsed{

        private final List<String> positionals = new ArrayList<>();
        private final Map<String, List<String>> values = new HashMap<>();
        private final Set<String> flags = new HashSet<>();

        public List<String> positionals(){
            return positionals;
        }

        public String value(String name){
            List<String> all = values.get(name);
            return all == null ? null : all.get(all.size() - 1);
        }

        public List<String> values(String name){
            List<String> all = values.get(name);
            return all == null ? new ArrayList<String>() : new ArrayList<>(all);
        }

        public boolean has(String flag){
            return flags.contains(flag);
        }
    }

    private final Set<String> valueOptions;
    private final Set<String> flagOptions;
    private final Set<String> repeatable;

    public ArgParser(String[] valueOptions, String[] flagOptions, String[] repeatable){
        this.valueOptions = new HashSet<>(Arrays.asList(valueOptions));
        this.flagOptions = new HashSet<>(Arrays.asList(flagOptions));
        this.repeatable = new HashSet<>(Arrays.asList(repeatable));
    }

    public Parsed parse(String[] args, int from){
        Parsed result = new Parsed();

        for(int i = from; i < args.length; i++){
            String arg = args[i];

            if(!arg.startsWith("--") || arg.length() == 2){
                result.positionals.add(arg);
                continue;
            }

            String name = arg.substring(2);
            String inlineValue = null;
            int equals = name.indexOf('=');
            if(equals >= 0){
                inlineValue = name.substring(equals + 1);
                name = name.substring(0, equals);
            }

            if(flagOptions.contains(name)){
                if(inlineValue != null){
                    throw new UsageException("the option --" + name + " does not take a value");
                }
                result.flags.add(name);
            }else if(valueOptions.contains(name)){
                String value = inlineValue;
                if(value == null){
                    if(i + 1 >= args.length){
                        throw new UsageException("the option --" + name + " needs a value");
                    }
                    value = args[++i];
                }
                if(result.values.containsKey(name) && !repeatable.contains(name)){
                    throw new UsageException("the option --" + name + " was given more than once");
                }
                result.values.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
            }else{
                List<String> known = new ArrayList<>(valueOptions);
                known.addAll(flagOptions);
                throw new UsageException("unknown option --" + name + Suggestions.hint(name, known));
            }
        }
        return result;
    }
}
