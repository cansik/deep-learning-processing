package ch.bildspur.deep;

import ch.bildspur.deep.learning.YoloResult;
import ch.bildspur.deep.learning.YoloDetector;
import processing.core.PApplet;
import processing.core.PImage;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

public class Application extends PApplet {

    PImage img;
    YoloDetector detector;
    List<YoloResult> results = new ArrayList<>();

    boolean modelLoaded = false;

    @Override
    public void settings() {
        size(800, 800, FX2D);
    }

    @Override
    public void setup() {
        img = loadImage("street.jpg");
        detector = new YoloDetector();
    }

    @Override
    public void draw() {
        background(0);

        // skip some frames
        if(frameCount < 5) {
            fill(255);
            textAlign(CENTER, CENTER);
            textSize(20);
            text("loading model...", width / 2, height / 2);
            return;
        }

        surface.setTitle("loading model...");

        // init yolo
        if(!modelLoaded) {
            detector.loadModel();
            modelLoaded = true;
        }

        // detect
        noLoop();
        image(img, 0, 0, width, height);

        Instant starts = Instant.now();
        try {
            surface.setTitle("detecting...");
            results = detector.detect(img, 0.5);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Instant ends = Instant.now();

        // show result
        strokeWeight(2f);
        float wr = 1.0f / detector.INPUT_WIDTH * width;
        float hr = 1.0f / detector.INPUT_HEIGHT * height;

        for(YoloResult result : results) {
            noFill();
            stroke(0, 255, 0);
            rect(result.getX() * wr, result.getY() * hr, result.getWidth() * wr, result.getHeight() * hr);

            fill(0, 255, 0);
            textAlign(LEFT, BOTTOM);
            textSize(8);
            text(result.getPredictedClassName(), result.getX() * wr, result.getY() * hr - 5);
        }

        // show infos
        surface.setTitle("Count: " + results.size() + " Time: " + Duration.between(starts, ends).toMillis() + " ms");
        saveFrame("data/result.jpg");
    }

    public void run() {
        runSketch();
    }
}
