package neuralforge.optimizer;

import neuralforge.schedule.StepDecay;
import static neuralforge.TestHelper.*;

public class OptimizerFactoryTest{

    public static void main(String[] args){
        testNames();
        testCreate();
        testDefaultRates();
        testWithSchedule();
        testErrors();
        finish();
    }

    private static void testNames(){
        String[] names = OptimizerFactory.names();
        check("five optimizers", names.length == 5);
        check("names are a copy", OptimizerFactory.names()[0].equals(names[0]));
        names[0] = "changed";
        check("changing the returned array changes nothing", OptimizerFactory.names()[0].equals("sgd"));
    }

    private static void testCreate(){
        check("sgd", OptimizerFactory.create("sgd") instanceof SGD);
        check("momentum", OptimizerFactory.create("momentum").name().startsWith("momentum("));
        check("nesterov", OptimizerFactory.create("nesterov").name().startsWith("nesterov("));
        check("rmsprop", OptimizerFactory.create("rmsprop") instanceof RMSProp);
        check("adam", OptimizerFactory.create("adam") instanceof Adam);

        check("not case sensitive", OptimizerFactory.create("  Adam ") instanceof Adam);
    }

    private static void testDefaultRates(){
        for(String name : OptimizerFactory.names()){
            double rate = OptimizerFactory.defaultRate(name);
            check(name + " has a positive default rate", rate > 0 && rate < 1);
            check(name + " starts with its default rate", near(OptimizerFactory.create(name).learningRate(), rate, 1e-15));
        }

        check("adam rate is smaller than the sgd rate", OptimizerFactory.defaultRate("adam") < OptimizerFactory.defaultRate("sgd"));
        check("momentum rate is smaller than the sgd rate", OptimizerFactory.defaultRate("momentum") < OptimizerFactory.defaultRate("sgd"));
    }

    private static void testWithSchedule(){
        Optimizer optimizer = OptimizerFactory.create("adam", new StepDecay(0.01, 0.5, 2));
        check("starts with the schedule's first rate", near(optimizer.learningRate(), 0.01, 1e-15));
        optimizer.startEpoch(2);
        check("follows the schedule", near(optimizer.learningRate(), 0.005, 1e-15));
        check("name shows the schedule", optimizer.name().contains("step("));
    }

    private static void testErrors(){
        expectThrows("unknown optimizer", () -> OptimizerFactory.create("adagrad"));
        expectThrows("unknown optimizer with schedule", () -> OptimizerFactory.create("lion", new StepDecay(0.1, 0.5, 2)));
        expectThrows("unknown default rate", () -> OptimizerFactory.defaultRate("adagrad"));
        expectThrows("null name", () -> OptimizerFactory.create(null));

        String message = neuralforge.TestData.messageOf(() -> OptimizerFactory.create("adagrad"));
        check("error lists the choices", message != null && message.contains("adam") && message.contains("sgd"));
    }
}
