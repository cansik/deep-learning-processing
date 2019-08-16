package ch.bildspur.deep.learning;

import org.deeplearning4j.nn.layers.objdetect.DetectedObject;

public class YoloResult {
    private DetectedObject detectedObject;
    private String predictedClassName = "";
    private float confidence;
    private float x1;
    private float y1;
    private float x2;
    private float y2;


    public YoloResult(DetectedObject detectedObject) {
        this.detectedObject = detectedObject;
        this.predictedClassName = YoloDetector.COCO_CLASSES[detectedObject.getPredictedClass()];

        // todo: find out why 32 is a good value
        float factor = 32;

        x1 = (float) detectedObject.getTopLeftXY()[0] * factor;
        y1 = (float) detectedObject.getTopLeftXY()[1] * factor;
        x2 = (float) detectedObject.getBottomRightXY()[0] * factor;
        y2 = (float) detectedObject.getBottomRightXY()[1] * factor;

        confidence = (float) detectedObject.getConfidence();
    }

    public String getPredictedClassName() {
        return predictedClassName;
    }

    public float getConfidence() {
        return confidence;
    }

    public float getX() {
        return x1;
    }

    public float getY() {
        return y1;
    }

    public float getWidth() {
        return x2 - x1;
    }

    public float getHeight() {
        return y2 - y1;
    }

    public DetectedObject getDetectedObject() {
        return detectedObject;
    }
}
