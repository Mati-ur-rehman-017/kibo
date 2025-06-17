// Create a new Java class, e.g., ObjectDetector.java in the same package

package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class ObjectDetector {
    private static final String TAG = "ObjectDetector";
    private Net net;
    private final List<String> classNames = new ArrayList<>();
    private final int inputWidth;
    private final int inputHeight;
    private final float confThreshold;
    private final float nmsThreshold;

    // Colors for drawing bounding boxes (optional)
    private final Scalar[] colors = {
            new Scalar(255, 0, 0), new Scalar(0, 255, 0), new Scalar(0, 0, 255),
            new Scalar(255, 255, 0), new Scalar(0, 255, 255), new Scalar(255, 0, 255)
    };

    public static class Detection {
        public int classId;
        public String className;
        public float confidence;
        public Rect box;

        public Detection(int classId, String className, float confidence, Rect box) {
            this.classId = classId;
            this.className = className;
            this.confidence = confidence;
            this.box = box;
        }

        @Override
        public String toString() {
            return "Detection{" +
                    "className='" + className + '\'' +
                    ", confidence=" + confidence +
                    ", box=" + box +
                    '}';
        }
    }

    public ObjectDetector(Context context, String modelName, String classNamesFile,
                          int inputWidth, int inputHeight,
                          float confThreshold, float nmsThreshold) throws IOException {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.confThreshold = confThreshold;
        this.nmsThreshold = nmsThreshold;

        String modelPath = getPath(modelName, context);
        String classesPath = getPath(classNamesFile, context); // Not strictly needed if reading directly

        if (modelPath == null) {
            throw new IOException("Failed to get path for model file: " + modelName);
        }

        // Load class names
        AssetManager assetManager = context.getAssets();
        try (InputStream is = assetManager.open(classNamesFile);
             Scanner scanner = new Scanner(is)) {
            while (scanner.hasNextLine()) {
                classNames.add(scanner.nextLine());
            }
            Log.i(TAG, "Loaded " + classNames.size() + " class names.");
        } catch (IOException e) {
            Log.e(TAG, "Error loading class names from " + classNamesFile, e);
            throw e;
        }

        // Load the network
        net = Dnn.readNetFromONNX(modelPath);
        if (net.empty()) {
            Log.e(TAG, "Failed to load ONNX model from: " + modelPath);
            throw new IOException("Failed to load ONNX model");
        } else {
            Log.i(TAG, "ONNX model loaded successfully from: " + modelPath);
        }

        // Set preferable backend and target (optional, but can improve performance)
        // net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV); // Default
        // net.setPreferableTarget(Dnn.DNN_TARGET_CPU);   // Default
        // For GPU acceleration if available and OpenCV is built with it:
        // net.setPreferableBackend(Dnn.DNN_BACKEND_CUDA); // Example for NVIDIA
        // net.setPreferableTarget(Dnn.DNN_TARGET_CUDA);
    }

    // Helper function to get absolute path from assets
    private String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        FileOutputStream outputStream = null;
        try {
            inputStream = new BufferedInputStream(assetManager.open(file));
            File outFile = new File(context.getCacheDir(), file);
            outputStream = new FileOutputStream(outFile);
            byte[] buffer = new byte[1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            Log.d(TAG, "Asset file " + file + " copied to: " + outFile.getAbsolutePath());
            return outFile.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, "Failed to get path for asset file: " + file, e);
        } finally {
            try {
                if (inputStream != null) inputStream.close();
                if (outputStream != null) outputStream.close();
            } catch (IOException e) {
                Log.e(TAG, "Error closing streams for asset file: " + file, e);
            }
        }
        return null;
    }


    public List<Detection> detect(Mat frame) {
        if (frame.empty()) {
            Log.w(TAG, "Input frame is empty. Cannot perform detection.");
            return Collections.emptyList();
        }

        Mat blob = Dnn.blobFromImage(frame, 1.0 / 255.0, new Size(inputWidth, inputHeight), new Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // YOLOv5 ONNX models typically have one output layer named "output0" or similar.
        // If you know the name, you can specify it. Otherwise, getUnconnectedOutLayersNames() can try to find it.
        List<String> outNames = getOutputLayerNames(net);
        List<Mat> outs = new ArrayList<>();
        net.forward(outs, outNames);


        // The output of YOLOv5 is typically [batch_size, num_detections, num_classes + 5]
        // For a single image, batch_size = 1.
        // num_detections can be large (e.g., 25200 for 640x640 input).
        // Each detection: [center_x, center_y, width, height, object_confidence, class1_score, class2_score, ...]
        Mat detectionsMat = outs.get(0); // Assuming single output layer
        // Reshape if necessary, sometimes it's (1, N, 5+C), other times (N, 5+C) after squeeze
        if (detectionsMat.dims() > 2) { // e.g. (1, 25200, 85)
            detectionsMat = detectionsMat.reshape(1, detectionsMat.size(1)); // (25200, 85)
        }


        List<Rect2d> boxesList = new ArrayList<>();
        List<Float> confidencesList = new ArrayList<>();
        List<Integer> classIdsList = new ArrayList<>();

        float frameWidth = frame.cols();
        float frameHeight = frame.rows();
        float xFactor = frameWidth / inputWidth;
        float yFactor = frameHeight / inputHeight;

        for (int i = 0; i < detectionsMat.rows(); ++i) {
            Mat row = detectionsMat.row(i);
            float objectConfidence = (float) row.get(0, 4)[0];

            if (objectConfidence >= this.confThreshold) {
                Mat classScores = row.colRange(5, detectionsMat.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(classScores);
                float classScore = (float) mm.maxVal;
                Point classIdPoint = mm.maxLoc;
                int classId = (int) classIdPoint.x; // classIdPoint.y will be 0 for a 1D Mat

                float finalConfidence = objectConfidence * classScore; // Some models output scores already multiplied

                if (finalConfidence >= this.confThreshold) {
                    float centerX = (float) row.get(0, 0)[0] * xFactor;
                    float centerY = (float) row.get(0, 1)[0] * yFactor;
                    float width = (float) row.get(0, 2)[0] * xFactor;
                    float height = (float) row.get(0, 3)[0] * yFactor;

                    double left = centerX - width / 2.0;
                    double top = centerY - height / 2.0;

                    boxesList.add(new Rect2d(left, top, width, height));
                    confidencesList.add(finalConfidence);
                    classIdsList.add(classId);
                }
            }
        }

        // Release intermediate Mats
        blob.release();
        for(Mat m : outs) m.release();
        detectionsMat.release();


        if (boxesList.isEmpty()) {
            return Collections.emptyList();
        }

        // Apply Non-Maximum Suppression
        MatOfRect2d boxes = new MatOfRect2d();
        boxes.fromList(boxesList);

        MatOfFloat confidences = new MatOfFloat();
        confidences.fromList(confidencesList);

        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        List<Detection> finalDetections = new ArrayList<>();
        int[]   indicesArray = indices.toArray();
        for (int idx : indicesArray) {
            Rect2d box = boxesList.get(idx);
            int classId = classIdsList.get(idx);
            float conf = confidencesList.get(idx);
            String className = (classId < classNames.size()) ? classNames.get(classId) : "Unknown";
            finalDetections.add(new Detection(classId, className, conf, new Rect((int)box.x, (int)box.y, (int)box.x + (int)box.width, (int)box.y + (int)box.height)));
        }

        // Release NMS Mats
        boxes.release();
        confidences.release();
        indices.release();

        return finalDetections;
    }

    public void drawDetections(Mat frame, List<Detection> detections) {
        if (detections == null || detections.isEmpty()) return;

        for (Detection detection : detections) {
            Rect box = detection.box;
            Scalar color = colors[detection.classId % colors.length];

            // Draw bounding box
            Imgproc.rectangle(frame, box.tl(), box.br(), color, 2);

            // Draw label
            String label = detection.className + ": " + String.format("%.2f", detection.confidence);
            int[] baseLine = new int[1];
            Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, 2, baseLine);

            Point labelTl = new Point(box.x, box.y - labelSize.height - baseLine[0] > 0 ? box.y - labelSize.height - baseLine[0] : box.y + labelSize.height);
            Point labelBr = new Point(labelTl.x + labelSize.width, labelTl.y + labelSize.height);

            Imgproc.rectangle(frame, labelTl, labelBr, color, Core.FILLED);
            Imgproc.putText(frame, label, new Point(labelTl.x, labelTl.y + labelSize.height - baseLine[0]/2 ), // Adjust Y for better text placement
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
        }
    }

    // Helper to get output layer names
    private List<String> getOutputLayerNames(Net net) {
        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();
        if (outLayers.isEmpty() && !layersNames.isEmpty()) {
            // Common for ONNX YOLOv5, output layer is often just "output0" or the last layer.
            // If getUnconnectedOutLayers is empty, try the last layer name, or a known default.
            // For many YOLOv5 ONNX models, it's "output0" or simply "output".
            // Let's try adding "output0" as a common default if others fail
            Log.w(TAG, "getUnconnectedOutLayers was empty. Trying 'output0'. Known layers: " + layersNames);
            if (layersNames.contains("output0")) names.add("output0");
            else if (layersNames.contains("output")) names.add("output");
            else if (!layersNames.isEmpty()) names.add(layersNames.get(layersNames.size() - 1)); // Fallback to last layer
        } else {
            for (int i : outLayers) {
                names.add(layersNames.get(i - 1)); // Layer IDs are 1-based.
            }
        }
        if (names.isEmpty()) {
            Log.e(TAG, "Could not determine output layer names. Detection will likely fail.");
        } else {
            Log.i(TAG, "Output layer names: " + names);
        }
        return names;
    }
}