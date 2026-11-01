package neuralforge.gui;

import neuralforge.activation.Softmax;
import neuralforge.layer.DenseLayer;
import neuralforge.math.Matrix;
import neuralforge.network.NeuralNetwork;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import static neuralforge.TestHelper.*;

public class DigitPanelTest{

    public static void main(String[] args){
        System.setProperty("java.awt.headless", "true");

        testPredictAndClear();
        testEmptyCanvas();
        testPainting();
        testErrors();
        finish();
    }

    private static DigitPanel panel(){
        return new DigitPanel(new DigitClassifier(DigitClassifierTest.pixelNetwork()), "test model");
    }

    private static void testPredictAndClear(){
        DigitPanel panel = panel();
        check("canvas has 4x4 cells", panel.getCanvas().size() == 4);
        check("starts with a question mark", panel.getDigitText().equals("?"));

        panel.getCanvas().stamp(3.5, 0.5);
        panel.predict();
        check("digit is shown", panel.getDigitText().equals("3"));
        check("confidence is shown", panel.getDetailText().contains("confidence"));
        check("top three are listed", panel.getDetailText().contains("1.") && panel.getDetailText().contains("2.") && panel.getDetailText().contains("3."));

        panel.clear();
        check("clear empties the canvas", panel.getCanvas().isEmpty());
        check("clear resets the answer", panel.getDigitText().equals("?") && !panel.getDetailText().contains("confidence"));
    }

    private static void testEmptyCanvas(){
        DigitPanel panel = panel();
        panel.predict();
        check("nothing drawn", panel.getDigitText().equals("?") && panel.getDetailText().contains("Nothing drawn"));
    }

    private static void testPainting(){
        DigitPanel panel = panel();
        panel.getCanvas().stamp(1.5, 1.5);

        Dimension size = panel.getPreferredSize();
        panel.setSize(size);
        panel.doLayout();
        BufferedImage image = new BufferedImage(size.width, size.height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        panel.printAll(g);
        g.dispose();
        check("panel has a size", size.width > 100 && size.height > 100);
    }

    private static void testErrors(){
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new DenseLayer(new Matrix(10, 3), new Matrix(1, 3), new Softmax()));
        expectThrows("inputs that are not a square", () -> new DigitPanel(new DigitClassifier(net), "x"));
    }
}
