package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.core.Mat;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.utils.Converters;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.calib3d.Calib3d;

import android.util.Log;
import gov.nasa.arc.astrobee.Kinematics;

import java.io.IOException; // Added
import java.util.List;
/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    private static final String TAG = "YourService"; // Added for logging
    private ObjectDetector yoloDetector;
    private static final String YOLO_MODEL_NAME = "model.onnx"; // YOUR MODEL FILE NAME
    private static final String YOLO_CLASSES_NAME = "custom.names";   // YOUR CLASSES FILE NAME
    private static final int YOLO_INPUT_WIDTH = 640; // Common for YOLOv5
    private static final int YOLO_INPUT_HEIGHT = 640;// Common for YOLOv5
    private static final float YOLO_CONF_THRESHOLD = 0.6f; // Adjust as needed
    private static final float YOLO_NMS_THRESHOLD = 0.65f; // Adjust as needed
    static {
        try {
            System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
            Log.i(TAG, "OpenCV loaded successfully.");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load OpenCV native library: " + e.getMessage());
        }
    }

    @Override
    protected void runPlan1(){
        // The mission starts.

        api.startMission();
        if (yoloDetector == null) {
            try {
                Log.i(TAG, "Initializing ObjectDetector...");
                yoloDetector = new ObjectDetector(this, YOLO_MODEL_NAME, YOLO_CLASSES_NAME,
                        YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT,
                        YOLO_CONF_THRESHOLD, YOLO_NMS_THRESHOLD);
                Log.i(TAG, "ObjectDetector initialized successfully.");
            } catch (IOException e) {
                Log.e(TAG, "Failed to initialize ObjectDetector", e);
//                api.notify QRCodeRecognizedFailed();
//                return; // Or try to continue without detection
            }
        }
        SymbolExtractor symbolExtractor = SymbolExtractor.create(api.getNavCamIntrinsics());
        // --- Area 1 ---
        Point point1 = new Point(10.9922d, -9.4623d, 5.2776d);
        Quaternion quat1 = new Quaternion(0f, 0f, -0.7071f, 0.7071f);
        api.moveTo(point1, quat1, false);
        Mat image1 = api.getMatNavCam();
        if(image1.empty()) {
            Log.e("YourService", "Failed to get image from NavCam at Area 1.");
        } else {
            api.saveMatImage(image1, "nav_cam_image_area1.png");
            Mat modifiedImage = symbolExtractor.extractSymbol(image1,101,false);
            api.saveMatImage(modifiedImage, "nav_cam_image_area1_modified.png");
            Log.i(TAG, "Performing detection on Area 1 image...");
            Mat imageForYolo = new Mat(); // Create a new Mat for the BGR image

            if (modifiedImage.channels() == 1) {
                Log.i(TAG, "modifiedImage is grayscale. Converting to BGR for YOLO.");
                Imgproc.cvtColor(modifiedImage, imageForYolo, Imgproc.COLOR_GRAY2RGB);
            }
            api.saveMatImage(imageForYolo,"testing_image.png");

            Mat imageToDetectOn = imageForYolo.clone(); // Clone for detection if original is needed
            List<ObjectDetector.Detection> detections = yoloDetector.detect(imageToDetectOn);
            Log.i(TAG, "Detections found: " + detections.size());

            imageToDetectOn.release(); // Release the clone used ONLY for input to detect() if it was just for that.

            Mat imageWithDetections = imageForYolo.clone();

            if (detections != null && !detections.isEmpty()) {
                Log.i(TAG, "Drawing " + detections.size() + " detections on the image.");
                yoloDetector.drawDetections(imageWithDetections, detections); // Call the drawDetections method

                // Save the image with detections
                String outputImageName = "testing_image_with_detections.png";
                try {
                    // Assuming api.saveMatImage takes (Mat image, String filename)
                    // and handles the necessary path conversion if needed (e.g., to external storage)
                    api.saveMatImage(imageWithDetections, outputImageName);
                    Log.i(TAG, "Image with detections saved as: " + outputImageName);
                } catch (Exception e) {
                    Log.e(TAG, "Error saving image with detections: " + outputImageName, e);
                }
            } else {
                Log.i(TAG, "No detections to draw.");
            }

            imageWithDetections.release();
            // Process detections
            String detectedItemName = null;
            int bb=1;
            for (ObjectDetector.Detection det : detections) {
                Log.i(TAG, "Area 1 Detection: " + det.toString());
                if ("diamond".equals(det.className) ||
                        "emerald".equals(det.className) ||
                        "crystal".equals(det.className)) {
                    continue; // Skip these target items
                }
                api.setAreaInfo(1,det.className,bb);
                bb=bb+1;
            }
            // api.setAreaInfo(1, "item_name", 1);
        }

        // --- Area 2 ---
        Point point2 = new Point(11.0528d, -8.97148d, 4.87973d);
        Quaternion quat2 = new Quaternion(0f, 0.707f, 0f, 0.707f);
        api.moveTo(point2, quat2, true);
        Mat image2 = api.getMatNavCam();
        if(image2.empty()) {
            Log.e("YourService", "Failed to get image from NavCam at Area 2.");
        } else {
            api.saveMatImage(image2, "nav_cam_image_area2.png");
            Mat modifiedImage2 = symbolExtractor.extractSymbol(image2,102,false);
            api.saveMatImage(modifiedImage2, "nav_cam_image_area2_modified.png");
            // api.setAreaInfo(2, "item_name", 1);
        }

        // --- Area 3 ---
        Point point3 = new Point(11.0106d, -7.8828d, 4.87863d);
        Quaternion quat3 = new Quaternion(0f, 0.707f, 0f, 0.707f);
        api.moveTo(point3, quat3, false);
        Mat image3 = api.getMatNavCam();
        if(image3.empty()) {
            Log.e("YourService", "Failed to get image from NavCam at Area 3.");
        } else {
            api.saveMatImage(image3, "nav_cam_image_area3.png");
            Mat modifiedImage3 = symbolExtractor.extractSymbol(image3,103,false);
            api.saveMatImage(modifiedImage3, "nav_cam_image_area3_modified.png");
            // api.setAreaInfo(3, "item_name", 1);
        }

        // --- Area 4 ---
        Point point4 = new Point(10.984684d, -6.8947d, 5.0276d);
        Quaternion quat4 = new Quaternion(0f, 0f, 1f, 0f);
        api.moveTo(point4, quat4, false);
        Mat image4 = api.getMatNavCam();
        if(image4.empty()) {
            Log.e("YourService", "Failed to get image from NavCam at Area 4.");
        } else {
            api.saveMatImage(image4, "nav_cam_image_area4.png");
            Mat modifiedImage4 = symbolExtractor.extractSymbol(image4,104,false);
            api.saveMatImage(modifiedImage4, "nav_cam_image_area4_modified.png");
            // api.setAreaInfo(4, "item_name", 1);
        }
        api.reportRoundingCompletion();

        // --- Astronaut Recognition ---
        Point point5 = new Point(11.193,-6.5107,4.9654);
        Quaternion quat5 = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(point5, quat5, false);
        Mat image5 = api.getMatNavCam();
        if(image5.empty()) {
            Log.e("YourService", "Failed to get image from NavCam at Astronaut Recognition.");
        } else {
            api.saveMatImage(image5, "nav_cam_image_astronaut_recognition.png");
            Mat modifiedImage5 = symbolExtractor.extractSymbol(image5,100,true);
            api.saveMatImage(modifiedImage5, "nav_cam_image_astronaut_recognition_modified.png");
            // api.setAreaInfo(5, "item_name", 1);
        }
        // Continue with the rest of your mission
        api.notifyRecognitionItem();
        api.takeTargetItemSnapshot();
    }

    @Override
    protected void runPlan2(){
       // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }
}