package neuralforge.optimizer;

import neuralforge.schedule.ConstantRate;
import neuralforge.schedule.LearningRateSchedule;
import java.util.Arrays;

public class OptimizerFactory{

    private static final String[] NAMES = {"sgd", "momentum", "nesterov", "rmsprop", "adam"};

    private OptimizerFactory(){
    }

    public static String[] names(){
        return NAMES.clone();
    }

    public static double defaultRate(String name){
        switch(normalize(name)){
            case "sgd":
                return 0.1;
            case "momentum":
            case "nesterov":
                return 0.02;
            case "rmsprop":
            case "adam":
                return 0.001;
            default:
                throw unknown(name);
        }
    }

    public static Optimizer create(String name){
        return create(name, new ConstantRate(defaultRate(name)));
    }

    public static Optimizer create(String name, LearningRateSchedule schedule){
        switch(normalize(name)){
            case "sgd":
                return new SGD(schedule);
            case "momentum":
                return new Momentum(schedule, 0.9, false);
            case "nesterov":
                return new Momentum(schedule, 0.9, true);
            case "rmsprop":
                return new RMSProp(schedule, 0.9, 1e-8);
            case "adam":
                return new Adam(schedule);
            default:
                throw unknown(name);
        }
    }

    private static String normalize(String name){
        return name == null ? "" : name.trim().toLowerCase();
    }

    private static IllegalArgumentException unknown(String name){
        return new IllegalArgumentException("unknown optimizer '" + name + "', choose one of " + Arrays.toString(NAMES));
    }
}
