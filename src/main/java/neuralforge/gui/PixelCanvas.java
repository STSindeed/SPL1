package neuralforge.gui;

public class PixelCanvas{

    private static final double RADIUS = 1.7;
    private static final double CORE = 0.6;

    private static final double STEP = 0.25;

    private final int size;

    private final double[] pixels;

    public PixelCanvas(int size){
        if(size <= 0){
            throw new IllegalArgumentException("size must be positive, got " + size);
        }
        this.size = size;
        this.pixels = new double[size * size];
    }

    public int size(){
        return size;
    }

    public double get(int row, int col){
        if(row < 0 || row >= size || col < 0 || col >= size){
            throw new IndexOutOfBoundsException("cell (" + row + ", " + col + ") is outside the " + size + "x" + size + " grid");
        }
        return pixels[row * size + col];
    }

    public void clear(){
        for(int i = 0; i < pixels.length; i++){
            pixels[i] = 0.0;
        }
    }

    public boolean isEmpty(){
        for(double value : pixels){
            if(value > 0.0){
                return false;
            }
        }
        return true;
    }

    public double[] getPixels(){
        return pixels.clone();
    }

    public void setPixels(double[] values){
        if(values.length != pixels.length){
            throw new IllegalArgumentException("expected " + pixels.length + " values but got " + values.length);
        }
        for(double value : values){
            if(!(value >= 0.0 && value <= 1.0)){
                throw new IllegalArgumentException("pixel values must be between 0 and 1, got " + value);
            }
        }
        System.arraycopy(values, 0, pixels, 0, pixels.length);
    }

    public void stamp(double x, double y){
        int firstCol = (int) Math.floor(x - RADIUS);
        int lastCol = (int) Math.ceil(x + RADIUS);
        int firstRow = (int) Math.floor(y - RADIUS);
        int lastRow = (int) Math.ceil(y + RADIUS);

        for(int row = Math.max(firstRow, 0); row <= Math.min(lastRow, size - 1); row++){
            for(int col = Math.max(firstCol, 0); col <= Math.min(lastCol, size - 1); col++){
                double dx = (col + 0.5) - x;
                double dy = (row + 0.5) - y;
                double distance = Math.sqrt(dx * dx + dy * dy);

                double value;
                if(distance <= CORE){
                    value = 1.0;
                }else if(distance >= RADIUS){
                    value = 0.0;
                }else{
                    value = (RADIUS - distance) / (RADIUS - CORE);
                }

                int index = row * size + col;
                if(value > pixels[index]){
                    pixels[index] = value;
                }
            }
        }
    }

    public void drawLine(double x0, double y0, double x1, double y1){
        double length = Math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        int steps = Math.max(1, (int) Math.ceil(length / STEP));
        for(int i = 0; i <= steps; i++){
            double t = (double) i / steps;
            stamp(x0 + t * (x1 - x0), y0 + t * (y1 - y0));
        }
    }
}
