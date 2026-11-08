package neuralforge.schedule;

import static neuralforge.TestHelper.*;

public class ScheduleTest{

    public static void main(String[] args){
        testConstant();
        testStep();
        testExponential();
        testInverseTime();
        testCosine();
        testAllDecayDownwards();
        testFactory();
        testErrors();
        finish();
    }

    private static void testConstant(){
        ConstantRate schedule = new ConstantRate(0.05);
        check("same in every epoch", schedule.rateAt(0) == 0.05 && schedule.rateAt(1) == 0.05 && schedule.rateAt(999) == 0.05);
        check("describe", schedule.describe().equals("constant"));
    }

    private static void testStep(){
        StepDecay schedule = new StepDecay(0.1, 0.5, 3);
        check("epochs 0..2", schedule.rateAt(0) == 0.1 && schedule.rateAt(2) == 0.1);
        check("epochs 3..5", near(schedule.rateAt(3), 0.05, 1e-15) && near(schedule.rateAt(5), 0.05, 1e-15));
        check("epoch 6", near(schedule.rateAt(6), 0.025, 1e-15));
        check("epoch 9", near(schedule.rateAt(9), 0.0125, 1e-15));
        check("describe", schedule.describe().startsWith("step("));
    }

    private static void testExponential(){
        ExponentialDecay schedule = new ExponentialDecay(0.1, 0.9);
        check("epoch 0 is the initial rate", near(schedule.rateAt(0), 0.1, 1e-15));
        check("epoch 1", near(schedule.rateAt(1), 0.09, 1e-15));
        check("epoch 2", near(schedule.rateAt(2), 0.081, 1e-15));
        check("epoch 10", near(schedule.rateAt(10), 0.1 * Math.pow(0.9, 10), 1e-15));

        check("gamma 1 is constant", new ExponentialDecay(0.1, 1.0).rateAt(50) == 0.1);
    }

    private static void testInverseTime(){
        InverseTimeDecay schedule = new InverseTimeDecay(0.1, 0.5);
        check("epoch 0", near(schedule.rateAt(0), 0.1, 1e-15));
        check("halved after 2 epochs", near(schedule.rateAt(2), 0.05, 1e-15));
        check("a third after 4 epochs", near(schedule.rateAt(4), 0.1 / 3.0, 1e-15));
        check("k 0 is constant", new InverseTimeDecay(0.1, 0.0).rateAt(30) == 0.1);
    }

    private static void testCosine(){
        CosineDecay schedule = new CosineDecay(0.1, 0.0, 10);
        check("starts at the initial rate", near(schedule.rateAt(0), 0.1, 1e-15));
        check("half way is half of the way down", near(schedule.rateAt(5), 0.05, 1e-12));
        check("ends at the minimum", near(schedule.rateAt(10), 0.0, 1e-15));
        check("stays at the minimum", schedule.rateAt(25) == 0.0);

        CosineDecay floor = new CosineDecay(0.1, 0.02, 10);
        check("halfway with a minimum", near(floor.rateAt(5), 0.06, 1e-12));
        check("minimum is kept", floor.rateAt(10) == 0.02 && floor.rateAt(11) == 0.02);

        double firstDrop = schedule.rateAt(0) - schedule.rateAt(1);
        double middleDrop = schedule.rateAt(4) - schedule.rateAt(5);
        check("steepest in the middle", middleDrop > firstDrop);
    }

    private static void testAllDecayDownwards(){
        LearningRateSchedule[] schedules = {
            new StepDecay(0.1, 0.5, 2), new ExponentialDecay(0.1, 0.8), new InverseTimeDecay(0.1, 0.3), new CosineDecay(0.1, 0.0, 20)
        };
        for(LearningRateSchedule schedule : schedules){
            boolean never = true;
            for(int e = 1; e < 30; e++){
                if(schedule.rateAt(e) > schedule.rateAt(e - 1)){
                    never = false;
                }
            }
            check(schedule.describe() + " never goes up", never);
        }
    }

    private static void testFactory(){
        int epochs = 20;
        for(String name : ScheduleFactory.names()){
            LearningRateSchedule schedule = ScheduleFactory.create(name, 0.1, epochs);
            check(name + " starts at the initial rate", near(schedule.rateAt(0), 0.1, 1e-15));
        }

        check("step ends lower", ScheduleFactory.create("step", 0.1, epochs).rateAt(epochs - 1) < 0.02);
        check("exp ends near 5%", near(ScheduleFactory.create("exp", 0.1, epochs).rateAt(epochs), 0.005, 1e-12));
        check("inverse ends at a tenth", near(ScheduleFactory.create("inverse", 0.1, epochs).rateAt(epochs), 0.01, 1e-12));
        check("cosine ends at 0", ScheduleFactory.create("cosine", 0.1, epochs).rateAt(epochs) == 0.0);
        check("constant stays", ScheduleFactory.create("constant", 0.1, epochs).rateAt(epochs - 1) == 0.1);

        check("upper case name", ScheduleFactory.create(" COSINE ", 0.1, 5).describe().startsWith("cosine"));
        check("one epoch", ScheduleFactory.create("step", 0.1, 1).rateAt(0) == 0.1);
        check("names are a copy", ScheduleFactory.names().length == 5);

        expectThrows("unknown schedule", () -> ScheduleFactory.create("linear", 0.1, 10));
        expectThrows("null name", () -> ScheduleFactory.create(null, 0.1, 10));
        expectThrows("zero epochs", () -> ScheduleFactory.create("step", 0.1, 0));
    }

    private static void testErrors(){
        expectThrows("constant 0", () -> new ConstantRate(0.0));
        expectThrows("constant negative", () -> new ConstantRate(-1.0));
        expectThrows("constant infinite", () -> new ConstantRate(Double.POSITIVE_INFINITY));
        expectThrows("constant NaN", () -> new ConstantRate(Double.NaN));
        expectThrows("negative epoch", () -> new ConstantRate(0.1).rateAt(-1));

        expectThrows("step factor 1", () -> new StepDecay(0.1, 1.0, 5));
        expectThrows("step factor 0", () -> new StepDecay(0.1, 0.0, 5));
        expectThrows("step every 0", () -> new StepDecay(0.1, 0.5, 0));
        expectThrows("step initial 0", () -> new StepDecay(0.0, 0.5, 5));
        expectThrows("step negative epoch", () -> new StepDecay(0.1, 0.5, 5).rateAt(-1));

        expectThrows("exp gamma above 1", () -> new ExponentialDecay(0.1, 1.1));
        expectThrows("exp gamma 0", () -> new ExponentialDecay(0.1, 0.0));
        expectThrows("exp negative epoch", () -> new ExponentialDecay(0.1, 0.9).rateAt(-1));

        expectThrows("inverse negative k", () -> new InverseTimeDecay(0.1, -0.1));
        expectThrows("inverse initial 0", () -> new InverseTimeDecay(0.0, 0.1));
        expectThrows("inverse negative epoch", () -> new InverseTimeDecay(0.1, 0.1).rateAt(-1));

        expectThrows("cosine minimum above initial", () -> new CosineDecay(0.1, 0.2, 10));
        expectThrows("cosine negative minimum", () -> new CosineDecay(0.1, -0.01, 10));
        expectThrows("cosine 0 epochs", () -> new CosineDecay(0.1, 0.0, 0));
        expectThrows("cosine negative epoch", () -> new CosineDecay(0.1, 0.0, 10).rateAt(-1));
    }
}
