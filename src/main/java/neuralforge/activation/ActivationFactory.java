package neuralforge.activation;

public class ActivationFactory{

    private ActivationFactory(){
    }

    public static Activation create(String name, double parameter){
        switch(name){
            case "relu":
                return new ReLU();
            case "leaky_relu":
                return new LeakyReLU(parameter);
            case "sigmoid":
                return new Sigmoid();
            case "tanh":
                return new Tanh();
            case "softmax":
                return new Softmax();
            case "linear":
                return new Linear();
            default:
                throw new IllegalArgumentException("unknown activation '" + name + "'");
        }
    }

    public static double parameterOf(Activation activation){
        if(activation instanceof LeakyReLU){
            return ((LeakyReLU) activation).getSlope();
        }
        return 0.0;
    }
}
