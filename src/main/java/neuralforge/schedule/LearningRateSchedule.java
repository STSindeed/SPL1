package neuralforge.schedule;

public interface LearningRateSchedule{

    double rateAt(int epoch);

    String describe();
}
