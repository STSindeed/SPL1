package neuralforge.gui;

import static neuralforge.TestHelper.*;

public class PixelCanvasTest{

    public static void main(String[] args){
        testStamp();
        testBlending();
        testEdges();
        testLine();
        testClearAndCopy();
        testSetPixels();
        testErrors();
        finish();
    }

    private static void testStamp(){
        PixelCanvas canvas = new PixelCanvas(28);
        check("starts empty", canvas.isEmpty());

        canvas.stamp(10.5, 10.5);
        check("not empty after a stamp", !canvas.isEmpty());
        check("centre is fully white", canvas.get(10, 10) == 1.0);

        double neighbour = canvas.get(10, 11);
        check("neighbour is grey", neighbour > 0.0 && neighbour < 1.0);
        check("symmetric left and right", near(canvas.get(10, 9), canvas.get(10, 11), 1e-12));
        check("symmetric up and down", near(canvas.get(9, 10), canvas.get(11, 10), 1e-12));

        check("far cell untouched", canvas.get(10, 13) == 0.0 && canvas.get(5, 5) == 0.0);

        boolean inRange = true;
        for(double value : canvas.getPixels()){
            if(value < 0.0 || value > 1.0){
                inRange = false;
            }
        }
        check("all values between 0 and 1", inRange);
    }

    private static void testBlending(){
        PixelCanvas canvas = new PixelCanvas(28);
        canvas.stamp(10.5, 10.5);
        double before = canvas.get(10, 12);

        canvas.stamp(11.5, 10.5);
        check("stamping again never darkens", canvas.get(10, 12) >= before);
        check("centre stays white", canvas.get(10, 10) == 1.0);
    }

    private static void testEdges(){
        PixelCanvas canvas = new PixelCanvas(28);
        canvas.stamp(0.0, 0.0);
        canvas.stamp(28.0, 28.0);
        canvas.stamp(-5.0, 14.0);
        canvas.stamp(100.0, 100.0);
        check("corner cell is drawn", canvas.get(0, 0) > 0.0 && canvas.get(27, 27) > 0.0);
        check("stamp far outside draws nothing", canvas.get(14, 0) == 0.0);
    }

    private static void testLine(){
        PixelCanvas canvas = new PixelCanvas(28);
        canvas.drawLine(3.5, 14.5, 24.5, 14.5);

        boolean complete = true;
        for(int col = 3; col <= 24; col++){
            if(canvas.get(14, col) != 1.0){
                complete = false;
            }
        }
        check("horizontal line has no gaps", complete);
        check("rows far from the line are empty", canvas.get(8, 10) == 0.0 && canvas.get(20, 10) == 0.0);

        PixelCanvas diagonal = new PixelCanvas(28);
        diagonal.drawLine(2.5, 2.5, 25.5, 25.5);
        boolean diagonalOk = true;
        for(int i = 2; i <= 25; i++){
            if(diagonal.get(i, i) != 1.0){
                diagonalOk = false;
            }
        }
        check("diagonal line has no gaps", diagonalOk);

        PixelCanvas dot = new PixelCanvas(28);
        dot.drawLine(7.5, 7.5, 7.5, 7.5);
        check("zero length line is a dot", dot.get(7, 7) == 1.0);
    }

    private static void testClearAndCopy(){
        PixelCanvas canvas = new PixelCanvas(28);
        canvas.stamp(5.5, 5.5);

        double[] copy = canvas.getPixels();
        check("copy has 784 values", copy.length == 784);
        copy[5 * 28 + 5] = 0.0;
        check("getPixels returns a copy", canvas.get(5, 5) == 1.0);

        canvas.clear();
        check("clear empties the canvas", canvas.isEmpty() && canvas.get(5, 5) == 0.0);
    }

    private static void testSetPixels(){
        PixelCanvas canvas = new PixelCanvas(2);
        canvas.setPixels(new double[]{0.0, 0.5, 1.0, 0.25});
        check("row by row order", canvas.get(0, 1) == 0.5 && canvas.get(1, 0) == 1.0 && canvas.get(1, 1) == 0.25);
        check("round trip", canvas.getPixels()[3] == 0.25);
    }

    private static void testErrors(){
        expectThrows("size 0", () -> new PixelCanvas(0));
        PixelCanvas canvas = new PixelCanvas(4);
        expectThrows("row outside", () -> canvas.get(4, 0));
        expectThrows("column outside", () -> canvas.get(0, -1));
        expectThrows("wrong number of values", () -> canvas.setPixels(new double[10]));
        expectThrows("value above 1", () -> canvas.setPixels(filled(16, 1.5)));
        expectThrows("negative value", () -> canvas.setPixels(filled(16, -0.1)));
        expectThrows("NaN value", () -> canvas.setPixels(filled(16, Double.NaN)));
    }

    private static double[] filled(int n, double value){
        double[] result = new double[n];
        java.util.Arrays.fill(result, value);
        return result;
    }
}
