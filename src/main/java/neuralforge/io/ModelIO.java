package neuralforge.io;

import neuralforge.activation.Activation;
import neuralforge.activation.ActivationFactory;
import neuralforge.layer.DenseLayer;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.zip.CRC32;
import java.util.zip.CheckedInputStream;
import java.util.zip.CheckedOutputStream;

public class ModelIO{

    private static final int MAGIC = 0x4E464D31;
    private static final int FORMAT_VERSION = 1;

    private static final int MAX_LAYERS = 1000;
    private static final long MAX_WEIGHTS_PER_LAYER = 50000000L;

    private ModelIO(){
    }

    public static void save(NeuralNetwork network, String path){
        List<DenseLayer> layers = network.getLayers();
        if(layers.isEmpty()){
            throw new IllegalArgumentException("the network has no layers, nothing to save");
        }
        checkFinite(layers);

        Path target = Paths.get(path);
        Path temp = Paths.get(path + ".tmp");

        try{
            Path folder = target.toAbsolutePath().getParent();
            if(folder != null){
                Files.createDirectories(folder);
            }

            try(OutputStream file = new BufferedOutputStream(Files.newOutputStream(temp))){
                CRC32 crc = new CRC32();
                DataOutputStream out = new DataOutputStream(new CheckedOutputStream(file, crc));

                writeBody(out, layers);

                out.writeInt((int) crc.getValue());
                out.flush();
            }
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING);
        }catch(IOException e){
            try{
                Files.deleteIfExists(temp);
            }catch(IOException ignored){
            }
            throw new UncheckedIOException("could not write " + path + ": " + e.getMessage(), e);
        }
    }

    public static NeuralNetwork load(String path){
        try(InputStream file = new BufferedInputStream(Files.newInputStream(Paths.get(path)))){
            CRC32 crc = new CRC32();
            DataInputStream in = new DataInputStream(new CheckedInputStream(file, crc));

            NeuralNetwork network = readBody(in, path);

            int calculated = (int) crc.getValue();
            int stored = in.readInt();
            if(calculated != stored){
                throw new IllegalArgumentException(path + " is damaged, its checksum does not match");
            }
            if(in.read() != -1){
                throw new IllegalArgumentException(path + " has extra data after the model");
            }
            return network;
        }catch(EOFException e){
            throw new IllegalArgumentException(path + " ended too early, the file is damaged or incomplete");
        }catch(IOException e){
            throw new UncheckedIOException("could not read " + path + ": " + e.getMessage(), e);
        }
    }

    private static void writeBody(DataOutputStream out, List<DenseLayer> layers) throws IOException{
        out.writeInt(MAGIC);
        out.writeInt(FORMAT_VERSION);
        out.writeInt(layers.size());

        for(DenseLayer layer : layers){
            Activation activation = layer.getActivation();
            out.writeInt(layer.inputSize());
            out.writeInt(layer.outputSize());
            out.writeUTF(activation.name());
            out.writeDouble(ActivationFactory.parameterOf(activation));

            Matrix weights = layer.getWeights();
            for(int i = 0; i < weights.rows(); i++){
                for(int j = 0; j < weights.cols(); j++){
                    out.writeDouble(weights.get(i, j));
                }
            }
            Matrix biases = layer.getBiases();
            for(int j = 0; j < biases.cols(); j++){
                out.writeDouble(biases.get(0, j));
            }
        }
    }

    private static NeuralNetwork readBody(DataInputStream in, String path) throws IOException{
        int magic = in.readInt();
        if(magic != MAGIC){
            throw new IllegalArgumentException(path + " is not a NeuralForge model file");
        }
        int version = in.readInt();
        if(version != FORMAT_VERSION){
            throw new IllegalArgumentException(path + " has format version " + version + ", this program reads version " + FORMAT_VERSION);
        }
        int layerCount = in.readInt();
        if(layerCount < 1 || layerCount > MAX_LAYERS){
            throw new IllegalArgumentException(path + " claims to have " + layerCount + " layers");
        }

        NeuralNetwork network = new NeuralNetwork();
        for(int k = 0; k < layerCount; k++){
            int inputSize = in.readInt();
            int outputSize = in.readInt();
            if(inputSize <= 0 || outputSize <= 0 || (long) inputSize * outputSize > MAX_WEIGHTS_PER_LAYER){
                throw new IllegalArgumentException(path + ": layer " + (k + 1) + " has an impossible size " + inputSize + " x " + outputSize);
            }

            String activationName = in.readUTF();
            double parameter = in.readDouble();
            Activation activation = ActivationFactory.create(activationName, parameter);

            Matrix weights = new Matrix(inputSize, outputSize);
            for(int i = 0; i < inputSize; i++){
                for(int j = 0; j < outputSize; j++){
                    weights.set(i, j, readFinite(in, path));
                }
            }
            Matrix biases = new Matrix(1, outputSize);
            for(int j = 0; j < outputSize; j++){
                biases.set(0, j, readFinite(in, path));
            }

            network.addLayer(new DenseLayer(weights, biases, activation));
        }
        return network;
    }

    private static double readFinite(DataInputStream in, String path) throws IOException{
        double value = in.readDouble();
        if(Double.isNaN(value) || Double.isInfinite(value)){
            throw new IllegalArgumentException(path + " contains a NaN or infinite value");
        }
        return value;
    }

    private static void checkFinite(List<DenseLayer> layers){
        for(int k = 0; k < layers.size(); k++){
            Matrix weights = layers.get(k).getWeights();
            Matrix biases = layers.get(k).getBiases();
            for(int i = 0; i < weights.rows(); i++){
                for(int j = 0; j < weights.cols(); j++){
                    if(!isFinite(weights.get(i, j))){
                        throw new IllegalArgumentException("layer " + (k + 1) + " has a NaN or infinite weight, training probably diverged");
                    }
                }
            }
            for(int j = 0; j < biases.cols(); j++){
                if(!isFinite(biases.get(0, j))){
                    throw new IllegalArgumentException("layer " + (k + 1) + " has a NaN or infinite bias, training probably diverged");
                }
            }
        }
    }

    private static boolean isFinite(double value){
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }
}
