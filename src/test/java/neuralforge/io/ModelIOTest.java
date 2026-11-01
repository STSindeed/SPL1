package neuralforge.io;

import neuralforge.TestData;
import neuralforge.activation.LeakyReLU;
import neuralforge.activation.Linear;
import neuralforge.activation.ReLU;
import neuralforge.activation.Softmax;
import neuralforge.activation.Tanh;
import neuralforge.init.XavierInitializer;
import neuralforge.layer.DenseLayer;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import static neuralforge.TestHelper.*;

public class ModelIOTest{

    public static void main(String[] args) throws IOException{
        testRoundTrip();
        testOutputsAreIdentical();
        testActivationsSurvive();
        testSaveBehaviour();
        testDamagedFiles();
        testWrongContent();
        testRefusesBadNetworks();
        finish();
    }

    private static NeuralNetwork sampleNetwork(){
        Random rng = new Random(5);
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(4, 6, new LeakyReLU(0.2), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(6, 5, new Tanh(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(5, 3, new Softmax(), new XavierInitializer(), rng));

        for(DenseLayer layer : net.getLayers()){
            for(int j = 0; j < layer.getBiases().cols(); j++){
                layer.getBiases().set(0, j, 0.1 * (j + 1));
            }
        }
        return net;
    }

    private static String tempPath(){
        String path = TestData.writeText(".nfm", "");
        new File(path).delete();
        new File(path).deleteOnExit();
        return path;
    }

    private static void testRoundTrip(){
        NeuralNetwork original = sampleNetwork();
        String path = tempPath();
        ModelIO.save(original, path);
        NeuralNetwork loaded = ModelIO.load(path);

        check("same number of layers", loaded.getLayers().size() == 3);
        check("same number of parameters", loaded.parameterCount() == original.parameterCount());

        boolean same = true;
        for(int k = 0; k < 3; k++){
            DenseLayer a = original.getLayers().get(k);
            DenseLayer b = loaded.getLayers().get(k);
            if(a.inputSize() != b.inputSize() || a.outputSize() != b.outputSize()){
                same = false;
            }
            if(!a.getActivation().name().equals(b.getActivation().name())){
                same = false;
            }

            if(!a.getWeights().approxEquals(b.getWeights(), 0.0) || !a.getBiases().approxEquals(b.getBiases(), 0.0)){
                same = false;
            }
        }
        check("sizes, activations, weights and biases are identical", same);
    }

    private static void testOutputsAreIdentical(){
        NeuralNetwork original = sampleNetwork();
        String path = tempPath();
        ModelIO.save(original, path);
        NeuralNetwork loaded = ModelIO.load(path);

        Matrix input = Matrix.randomUniform(20, 4, -2, 2, new Random(8));
        check("loaded network gives exactly the same output", original.forward(input).approxEquals(loaded.forward(input), 0.0));
    }

    private static void testActivationsSurvive(){
        NeuralNetwork net = new NeuralNetwork();
        Random rng = new Random(1);
        net.addLayer(new DenseLayer(2, 3, new LeakyReLU(0.3), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(3, 3, new ReLU(), new XavierInitializer(), rng));
        net.addLayer(new DenseLayer(3, 2, new Linear(), new XavierInitializer(), rng));

        String path = tempPath();
        ModelIO.save(net, path);
        NeuralNetwork loaded = ModelIO.load(path);

        check("activation names", loaded.getLayers().get(0).getActivation().name().equals("leaky_relu")
            && loaded.getLayers().get(1).getActivation().name().equals("relu")
            && loaded.getLayers().get(2).getActivation().name().equals("linear"));

        Matrix negative = new Matrix(new double[][]{{-1.0, -2.0}});
        check("leaky slope is kept", net.forward(negative).approxEquals(loaded.forward(negative), 0.0));
    }

    private static void testSaveBehaviour() throws IOException{
        String folder = Files.createTempDirectory("neuralforge").toString();
        String path = folder + "/models/deep/model.nfm";
        new File(path).deleteOnExit();
        new File(folder + "/models/deep").deleteOnExit();
        new File(folder + "/models").deleteOnExit();
        new File(folder).deleteOnExit();

        ModelIO.save(sampleNetwork(), path);
        check("file is created in a new folder", new File(path).exists());
        check("no temporary file is left", !new File(path + ".tmp").exists());

        NeuralNetwork other = new NeuralNetwork();
        other.addLayer(new DenseLayer(2, 2, new Softmax(), new XavierInitializer(), new Random(3)));
        ModelIO.save(other, path);
        check("old file is replaced", ModelIO.load(path).getLayers().size() == 1);
    }

    private static void testDamagedFiles() throws IOException{
        String path = tempPath();
        ModelIO.save(sampleNetwork(), path);
        byte[] good = Files.readAllBytes(Paths.get(path));

        byte[] flipped = good.clone();
        flipped[good.length / 2] ^= 0x10;
        String flippedPath = TestData.writeBytes(".nfm", flipped);
        String message = TestData.messageOf(() -> ModelIO.load(flippedPath));
        check("flipped byte is caught by the checksum", message != null && message.contains("damaged"));

        byte[] half = new byte[good.length / 2];
        System.arraycopy(good, 0, half, 0, half.length);
        String halfPath = TestData.writeBytes(".nfm", half);
        String cut = TestData.messageOf(() -> ModelIO.load(halfPath));
        check("cut file", cut != null && cut.contains("too early"));

        byte[] longer = new byte[good.length + 1];
        System.arraycopy(good, 0, longer, 0, good.length);
        String longerPath = TestData.writeBytes(".nfm", longer);
        String extra = TestData.messageOf(() -> ModelIO.load(longerPath));
        check("extra byte", extra != null && extra.contains("extra data"));

        expectThrows("empty file", () -> ModelIO.load(TestData.writeBytes(".nfm", new byte[0])));
        expectThrows("file does not exist", () -> ModelIO.load("no/such/model.nfm"));
    }

    private static void testWrongContent(){
        String text = TestData.messageOf(() -> ModelIO.load(TestData.writeText(".nfm", "this is not a model at all")));
        check("wrong magic number", text != null && text.contains("not a NeuralForge model"));

        String path = tempPath();
        ModelIO.save(sampleNetwork(), path);
        try{
            byte[] bytes = Files.readAllBytes(Paths.get(path));
            bytes[7] = 2;
            String future = TestData.writeBytes(".nfm", bytes);
            String message = TestData.messageOf(() -> ModelIO.load(future));
            check("other format version", message != null && message.contains("version 2"));

            byte[] noLayers = Files.readAllBytes(Paths.get(path));
            noLayers[11] = 0;
            String noLayersPath = TestData.writeBytes(".nfm", noLayers);
            expectThrows("no layers", () -> ModelIO.load(noLayersPath));

            byte[] huge = Files.readAllBytes(Paths.get(path));
            huge[12] = 0x7f;
            String hugePath = TestData.writeBytes(".nfm", huge);
            expectThrows("impossible layer size", () -> ModelIO.load(hugePath));
        }catch(IOException e){
            check("could not read the model back: " + e.getMessage(), false);
        }
    }

    private static void testRefusesBadNetworks(){
        expectThrows("empty network", () -> ModelIO.save(new NeuralNetwork(), tempPath()));

        NeuralNetwork nan = sampleNetwork();
        nan.getLayers().get(1).getWeights().set(2, 2, Double.NaN);
        String nanPath = tempPath();
        expectThrows("NaN weight", () -> ModelIO.save(nan, nanPath));
        check("nothing written for a bad network", !new File(nanPath).exists());

        NeuralNetwork infinite = sampleNetwork();
        infinite.getLayers().get(0).getBiases().set(0, 0, Double.POSITIVE_INFINITY);
        expectThrows("infinite bias", () -> ModelIO.save(infinite, tempPath()));
    }
}
