package ch.bildspur.deep.learning;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import processing.core.PImage;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class YoloDetector {
    protected static final String[] COCO_CLASSES = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
            "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush" };

    public final int INPUT_WIDTH = 608;
    public final int INPUT_HEIGHT = 608;
    private final int INPUT_CHANNELS = 3;
    private final String MODEL_FILE_NAME = "data/saved_model";

    private ComputationGraph yoloModel;
    private NativeImageLoader imageLoader;

    public void loadModel() {
        try {
            File modelFile = new File(MODEL_FILE_NAME);
            if (!modelFile.exists()) {
                System.out.println("Cached model does NOT exists, initializing...");
                yoloModel = (ComputationGraph) YOLO2.builder().build().initPretrained();
                yoloModel.save(modelFile);
            } else {
                System.out.println("Cached model does exists");
                yoloModel = ModelSerializer.restoreComputationGraph(modelFile);
            }

            imageLoader = new NativeImageLoader(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
        } catch (IOException e) {
            throw new Error("Not able to init the model", e);
        }
    }

    public List<YoloResult> detect(PImage img, double threshold) throws IOException {
        INDArray mat = convertToMat(img);

        long start = System.currentTimeMillis();
        INDArray output = yoloModel.outputSingle(mat);
        long end = System.currentTimeMillis();
        System.out.println("simple forward took: " + (end - start) + " ms");

        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) yoloModel.getOutputLayer(0);
        List<DetectedObject> predictedObjects = outputLayer.getPredictedObjects(output, threshold);

        return predictedObjects.stream().map(YoloResult::new).collect(Collectors.toList());
    }

    private INDArray convertToMat(PImage img) throws IOException {
        BufferedImage m = (BufferedImage) img.getNative();
        INDArray image = imageLoader.asMatrix(m);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        return image;
    }
}
