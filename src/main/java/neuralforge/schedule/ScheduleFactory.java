package neuralforge.schedule;

import java.util.Arrays;

public class ScheduleFactory{

    private static final String[] NAMES = {"constant", "step", "exp", "inverse", "cosine"};

    private ScheduleFactory(){
    }

    public static String[] names(){
        return NAMES.clone();
    }

    public static LearningRateSchedule create(String name, double initial, int epochs){
        if(epochs <= 0){
            throw new IllegalArgumentException("epochs must be positive, got " + epochs);
        }
        String key = name == null ? "" : name.trim().toLowerCase();
        switch(key){
            case "constant":
                return new ConstantRate(initial);
            case "step":
                return new StepDecay(initial, 0.5, Math.max(1, epochs / 4));
            case "exp":
                return new ExponentialDecay(initial, Math.pow(0.05, 1.0 / epochs));
            case "inverse":
                return new InverseTimeDecay(initial, 9.0 / epochs);
            case "cosine":
                return new CosineDecay(initial, 0.0, epochs);
            default:
                throw new IllegalArgumentException("unknown schedule '" + name + "', choose one of " + Arrays.toString(NAMES));
        }
    }
}
