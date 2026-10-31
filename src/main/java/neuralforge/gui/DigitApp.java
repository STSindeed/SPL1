package neuralforge.gui;

import neuralforge.io.ModelIO;
import neuralforge.network.NeuralNetwork;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import java.awt.GraphicsEnvironment;

public class DigitApp{

    private DigitApp(){
    }

    public static void main(String[] args){
        launch(args.length > 0 ? args[0] : "models/mnist.nfm");
    }

    public static void launch(String modelPath){
        if(GraphicsEnvironment.isHeadless()){
            System.out.println("this computer has no display, the window cannot be opened");
            return;
        }

        NeuralNetwork network = ModelIO.load(modelPath);
        DigitClassifier classifier = new DigitClassifier(network);
        String info = "model " + modelPath + "  (" + classifier.describe() + ")";

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("NeuralForge digit recognizer");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setContentPane(new DigitPanel(classifier, info));
            frame.pack();
            frame.setResizable(false);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
