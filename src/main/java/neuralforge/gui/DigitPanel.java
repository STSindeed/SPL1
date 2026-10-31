package neuralforge.gui;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.util.Locale;

public class DigitPanel extends JPanel{

    private static final long serialVersionUID = 1L;

    private static final int CELL_SIZE = 10;
    private static final int TOP_COUNT = 3;

    private final DigitClassifier classifier;
    private final PixelCanvas canvas;
    private final DrawingPanel drawing;

    private final JLabel digitLabel = new JLabel("?", JLabel.CENTER);
    private final JLabel detailLabel = new JLabel();

    public DigitPanel(DigitClassifier classifier, String modelInfo){
        this.classifier = classifier;

        int side = (int) Math.round(Math.sqrt(classifier.inputSize()));
        if(side * side != classifier.inputSize()){
            throw new IllegalArgumentException("the network takes " + classifier.inputSize() + " inputs, that is not a square picture");
        }
        this.canvas = new PixelCanvas(side);
        this.drawing = new DrawingPanel(canvas, CELL_SIZE);

        setLayout(new BorderLayout(12, 12));
        setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));

        JLabel title = new JLabel("Draw a digit with the mouse, then press Predict");
        add(title, BorderLayout.NORTH);

        JPanel drawingHolder = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));
        drawing.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        drawingHolder.add(drawing);
        add(drawingHolder, BorderLayout.CENTER);

        add(buildResultPanel(), BorderLayout.EAST);
        add(buildBottomPanel(modelInfo), BorderLayout.SOUTH);

        resetResult();
    }

    private JPanel buildResultPanel(){
        JPanel panel = new JPanel(new BorderLayout(0, 8));
        panel.setPreferredSize(new Dimension(190, 0));
        panel.setBorder(BorderFactory.createTitledBorder("Prediction"));

        digitLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 96));
        panel.add(digitLabel, BorderLayout.CENTER);

        detailLabel.setVerticalAlignment(JLabel.TOP);
        detailLabel.setPreferredSize(new Dimension(170, 90));
        panel.add(detailLabel, BorderLayout.SOUTH);
        return panel;
    }

    private JPanel buildBottomPanel(String modelInfo){
        JPanel bottom = new JPanel(new BorderLayout(8, 0));

        JLabel info = new JLabel(modelInfo);
        info.setForeground(Color.GRAY);
        bottom.add(info, BorderLayout.CENTER);

        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> clear());

        JButton predictButton = new JButton("Predict");
        predictButton.addActionListener(e -> predict());

        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        buttons.add(clearButton);
        buttons.add(predictButton);
        bottom.add(buttons, BorderLayout.EAST);
        return bottom;
    }

    public PixelCanvas getCanvas(){
        return canvas;
    }

    public void clear(){
        canvas.clear();
        drawing.repaint();
        resetResult();
    }

    public void predict(){
        if(canvas.isEmpty()){
            digitLabel.setText("?");
            detailLabel.setText("<html>Nothing drawn yet.</html>");
            return;
        }

        Prediction prediction = classifier.predict(canvas.getPixels());
        digitLabel.setText(String.valueOf(prediction.digit()));

        StringBuilder text = new StringBuilder("<html>");
        text.append(String.format(Locale.US, "confidence %.1f%%<br><br>", 100 * prediction.confidence()));
        int[] top = prediction.top(Math.min(TOP_COUNT, classifier.classCount()));
        for(int rank = 0; rank < top.length; rank++){
            text.append(String.format(Locale.US, "%d.  digit %d   %.1f%%<br>", rank + 1, top[rank], 100 * prediction.probability(top[rank])));
        }
        text.append("</html>");
        detailLabel.setText(text.toString());
    }

    public String getDigitText(){
        return digitLabel.getText();
    }

    public String getDetailText(){
        return detailLabel.getText();
    }

    private void resetResult(){
        digitLabel.setText("?");
        detailLabel.setText("<html>Press Predict after drawing.</html>");
    }
}
