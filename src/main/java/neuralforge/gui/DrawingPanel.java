package neuralforge.gui;

import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class DrawingPanel extends JPanel{

    private static final long serialVersionUID = 1L;

    private final PixelCanvas canvas;
    private final int cellSize;

    private double lastX;
    private double lastY;

    public DrawingPanel(PixelCanvas canvas, int cellSize){
        this.canvas = canvas;
        this.cellSize = cellSize;

        setPreferredSize(new Dimension(canvas.size() * cellSize, canvas.size() * cellSize));
        setBackground(Color.BLACK);
        setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));

        MouseAdapter mouse = new MouseAdapter(){
            @Override
            public void mousePressed(MouseEvent e){
                if(!SwingUtilities.isLeftMouseButton(e)){
                    return;
                }
                lastX = toCells(e.getX());
                lastY = toCells(e.getY());
                canvas.stamp(lastX, lastY);
                repaint();
            }

            @Override
            public void mouseDragged(MouseEvent e){
                if(!SwingUtilities.isLeftMouseButton(e)){
                    return;
                }
                double x = toCells(e.getX());
                double y = toCells(e.getY());
                canvas.drawLine(lastX, lastY, x, y);
                lastX = x;
                lastY = y;
                repaint();
            }
        };
        addMouseListener(mouse);
        addMouseMotionListener(mouse);
    }

    private double toCells(int pixel){
        return (double) pixel / cellSize;
    }

    @Override
    protected void paintComponent(Graphics g){
        super.paintComponent(g);

        for(int row = 0; row < canvas.size(); row++){
            for(int col = 0; col < canvas.size(); col++){
                double value = canvas.get(row, col);
                if(value > 0.0){
                    int gray = (int) Math.round(255 * value);
                    g.setColor(new Color(gray, gray, gray));
                    g.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
                }
            }
        }
    }
}
