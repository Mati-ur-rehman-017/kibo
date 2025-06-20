package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Kinematics;

import org.opencv.core.Mat;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.core.CvType;


import java.util.List;
import org.opencv.imgproc.Imgproc;
import android.util.Log;
import java.io.IOException;
import java.lang.invoke.WrongMethodTypeException;
import java.util.Arrays; // Needed for Arrays.toString() and Arrays.asList()
import java.util.ArrayList;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    private static final String TAG = "YourService";
    private ObjectDetector yoloDetector;
    private static final String YOLO_MODEL_NAME = "model.onnx";
    private static final String YOLO_CLASSES_NAME = "custom.names";
    private static final int YOLO_INPUT_WIDTH = 640;
    private static final int YOLO_INPUT_HEIGHT = 640;
    private static final float YOLO_CONF_THRESHOLD = 0.6f;
    private static final float YOLO_NMS_THRESHOLD = 0.65f;
    private static SymbolExtractor symbolExtractor;
    private static Mat cameraMatrix;
    private static Mat distCoeffs;
    private static WorldPose[] finalLocations = new WorldPose[4];
    private static WorldPose[] locations = new WorldPose[4];

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
                return;
            }
        }
        symbolExtractor = SymbolExtractor.create(api.getNavCamIntrinsics());

        intrinsicsToMat(api.getNavCamIntrinsics(), true);
        String[] targetItemsInArea = new String[4];

        locations[0] = new WorldPose(new Point(10.9922d, -9.4623d, 5.2776d),new Quaternion(0f, 0f, -0.7071f, 0.7071f));
        locations[1] = new WorldPose(new Point(11.0106d, -8.875d, 4.66203d), new Quaternion(0.5f, 0.5f, -0.5f, 0.5f));
        locations[2] = new WorldPose(new Point(11.0106d, -7.95d, 4.66093d), new Quaternion(0.5f, 0.5f, -0.5f, 0.5f));
        locations[3] = new WorldPose(new Point(10.984684d, -6.8947d, 5.0276d), new Quaternion(0f, 0f, 1f, 0f));
        // --- Area 1 ---
        Integer[] ids = {101,102,103,104};
        moveTo(locations[0]);
        Mat imageArea1 = extractSymbolFromCam(false, ids[0]);
        if(imageArea1.empty()) {
            Log.e(TAG, "Failed to get image from NavCam at Area 1.");
        } else {
            Mat imageForYolo1 = new Mat();
            if (imageArea1.channels() == 1) {
                Imgproc.cvtColor(imageArea1, imageForYolo1, Imgproc.COLOR_GRAY2RGB);
            } else {
                imageArea1.copyTo(imageForYolo1);
            }
            api.saveMatImage(imageForYolo1, "nav_cam_image_area1_yolo_input.png");

            Mat imageToDetectOn1 = imageForYolo1.clone();
            List<ObjectDetector.Detection> detections1 = yoloDetector.detect(imageToDetectOn1);
            Log.i(TAG, "Area 1: Detections found: " + (detections1 != null ? detections1.size() : 0));
            imageToDetectOn1.release();

            if (detections1 != null && !detections1.isEmpty()) {
                Mat imageWithDetections1 = imageForYolo1.clone();
                yoloDetector.drawDetections(imageWithDetections1, detections1);
                api.saveMatImage(imageWithDetections1, "nav_cam_image_area1_detections.png");
                imageWithDetections1.release();

                int itemIndexForSetAreaInfo1 = 1;
                for (ObjectDetector.Detection det : detections1) {
                    Log.i(TAG, "Area 1 Processing Detection: " + det.toString());
                    if ("diamond".equals(det.className) ||
                            "emerald".equals(det.className) ||
                            "crystal".equals(det.className)) {
                        targetItemsInArea[0] = det.className;
                        Log.i(TAG, "Area 1: Target class item '" + det.className + "' identified.");
                    } else {
                        api.setAreaInfo(1, det.className, itemIndexForSetAreaInfo1);
                        itemIndexForSetAreaInfo1++;
                    }
                }
            }
            imageForYolo1.release();
            imageArea1.release();
        }

        // --- Area 2 ---
        moveTo(locations[1]);
        Mat imageArea2 = extractSymbolFromCam(false, ids[1]);
        if(imageArea2.empty()) {
            Log.e(TAG, "Failed to get image from NavCam at Area 2.");
        } else {
            Mat imageForYolo2 = new Mat();
            if (imageArea2.channels() == 1) {
                Imgproc.cvtColor(imageArea2, imageForYolo2, Imgproc.COLOR_GRAY2RGB);
            } else {
                imageArea2.copyTo(imageForYolo2);
            }
            api.saveMatImage(imageForYolo2, "nav_cam_image_area2_yolo_input.png");

            Mat imageToDetectOn2 = imageForYolo2.clone();
            List<ObjectDetector.Detection> detections2 = yoloDetector.detect(imageToDetectOn2);
            Log.i(TAG, "Area 2: Detections found: " + (detections2 != null ? detections2.size() : 0));
            imageToDetectOn2.release();

            if (detections2 != null && !detections2.isEmpty()) {
                Mat imageWithDetections2 = imageForYolo2.clone();
                yoloDetector.drawDetections(imageWithDetections2, detections2);
                api.saveMatImage(imageWithDetections2, "nav_cam_image_area2_detections.png");
                imageWithDetections2.release();

                int itemIndexForSetAreaInfo2 = 1;
                for (ObjectDetector.Detection det : detections2) {
                    Log.i(TAG, "Area 2 Processing Detection: " + det.toString());
                    if ("diamond".equals(det.className) ||
                            "emerald".equals(det.className) ||
                            "crystal".equals(det.className)) {
                        targetItemsInArea[1] = det.className;
                        Log.i(TAG, "Area 2: Target class item '" + det.className + "' identified.");
                    } else {
                        api.setAreaInfo(2, det.className, itemIndexForSetAreaInfo2);
                        itemIndexForSetAreaInfo2++;
                    }
                }
            }
            imageForYolo2.release();
            imageArea2.release();
        }

        // --- Area 3 ---
        moveTo(locations[2]);
        Mat imageArea3 = extractSymbolFromCam(false, ids[2]);
        if(imageArea3.empty()) {
            Log.e(TAG, "Failed to get image from NavCam at Area 3.");
        } else {
            Mat imageForYolo3 = new Mat();
            if (imageArea3.channels() == 1) {
                Imgproc.cvtColor(imageArea3, imageForYolo3, Imgproc.COLOR_GRAY2RGB);
            } else {
                imageArea3.copyTo(imageForYolo3);
            }
            api.saveMatImage(imageForYolo3, "nav_cam_image_area3_yolo_input.png");

            Mat imageToDetectOn3 = imageForYolo3.clone();
            List<ObjectDetector.Detection> detections3 = yoloDetector.detect(imageToDetectOn3);
            Log.i(TAG, "Area 3: Detections found: " + (detections3 != null ? detections3.size() : 0));
            imageToDetectOn3.release();

            if (detections3 != null && !detections3.isEmpty()) {
                Mat imageWithDetections3 = imageForYolo3.clone();
                yoloDetector.drawDetections(imageWithDetections3, detections3);
                api.saveMatImage(imageWithDetections3, "nav_cam_image_area3_detections.png");
                imageWithDetections3.release();

                int itemIndexForSetAreaInfo3 = 1;
                for (ObjectDetector.Detection det : detections3) {
                    Log.i(TAG, "Area 3 Processing Detection: " + det.toString());
                    if ("diamond".equals(det.className) ||
                            "emerald".equals(det.className) ||
                            "crystal".equals(det.className)) {
                        targetItemsInArea[2] = det.className;
                        Log.i(TAG, "Area 3: Target class item '" + det.className + "' identified.");
                    } else {
                        api.setAreaInfo(3, det.className, itemIndexForSetAreaInfo3);
                        itemIndexForSetAreaInfo3++;
                    }
                }
            }
            imageForYolo3.release();
            imageArea3.release();
        }

        // --- Area 4 ---
        moveTo(locations[3]);
        Mat imageArea4 = extractSymbolFromCam(false, ids[3]);
        if(imageArea4.empty()) {
            Log.e(TAG, "Failed to get image from NavCam at Area 4.");
        } else {
            Mat imageForYolo4 = new Mat();
            if (imageArea4.channels() == 1) {
                Imgproc.cvtColor(imageArea4, imageForYolo4, Imgproc.COLOR_GRAY2RGB);
            } else {
                imageArea4.copyTo(imageForYolo4);
            }
            api.saveMatImage(imageForYolo4, "nav_cam_image_area4_yolo_input.png");

            Mat imageToDetectOn4 = imageForYolo4.clone();
            List<ObjectDetector.Detection> detections4 = yoloDetector.detect(imageToDetectOn4);
            Log.i(TAG, "Area 4: Detections found: " + (detections4 != null ? detections4.size() : 0));
            imageToDetectOn4.release();

            if (detections4 != null && !detections4.isEmpty()) {
                Mat imageWithDetections4 = imageForYolo4.clone();
                yoloDetector.drawDetections(imageWithDetections4, detections4);
                api.saveMatImage(imageWithDetections4, "nav_cam_image_area4_detections.png");
                imageWithDetections4.release();

                int itemIndexForSetAreaInfo4 = 1;
                for (ObjectDetector.Detection det : detections4) {
                    Log.i(TAG, "Area 4 Processing Detection: " + det.toString());
                    if ("diamond".equals(det.className) ||
                            "emerald".equals(det.className) ||
                            "crystal".equals(det.className)) {
                        targetItemsInArea[3] = det.className;
                        Log.i(TAG, "Area 4: Target class item '" + det.className + "' identified.");
                    } else {
                        api.setAreaInfo(4, det.className, itemIndexForSetAreaInfo4);
                        itemIndexForSetAreaInfo4++;
                    }
                }
            }
            imageForYolo4.release();
            imageArea4.release();
        }

        Log.i(TAG, "Target class items identified in areas 1-4: " + Arrays.toString(targetItemsInArea));
        restrict();
        api.reportRoundingCompletion();

        // --- Astronaut Recognition ---
        String astronautSelectedItem = null; // Variable to store the item name selected by astronaut
        WorldPose astronautLocation = new WorldPose(new Point(11.193,-6.5107,4.9654), new Quaternion(0f, 0f, 0.707f, 0.7f));
        moveTo(astronautLocation);
        Mat targetImage = extractSymbolFromCam(true, 100);

        if (targetImage.empty()) {
            Log.e(TAG, "Astronaut Recognition: Failed to get image from NavCam.");
        } else {
            if (targetImage == null || targetImage.empty()) {
                Log.e(TAG, "Astronaut Recognition: targetImage is null or empty after symbol extraction.");
            } else {
                Mat imageForYolo5 = new Mat();
                if (targetImage.channels() == 1) {
                    Imgproc.cvtColor(targetImage, imageForYolo5, Imgproc.COLOR_GRAY2RGB);
                } else {
                    targetImage.copyTo(imageForYolo5);
                }
                api.saveMatImage(imageForYolo5, "nav_cam_image_astronaut_yolo_input.png");

                Mat imageToDetectOn5 = imageForYolo5.clone();
                List<ObjectDetector.Detection> detections5 = yoloDetector.detect(imageToDetectOn5);
                Log.i(TAG, "Astronaut Recognition: Detections on modified image: " + (detections5 != null ? detections5.size() : 0));
                imageToDetectOn5.release();

                if (detections5 != null && !detections5.isEmpty()) {
                    ObjectDetector.Detection localBestAstronautDetection = null; // Renamed to avoid confusion, local to this block
                    float maxPrimaryConf = 0f;
                    for (ObjectDetector.Detection det : detections5) {
                        if (("diamond".equals(det.className) ||
                                "emerald".equals(det.className) ||
                                "crystal".equals(det.className)) && det.confidence > maxPrimaryConf) {
                            localBestAstronautDetection = det;
                            maxPrimaryConf = det.confidence;
                        }
                    }

                    if (localBestAstronautDetection != null) {
                        astronautSelectedItem = localBestAstronautDetection.className;
                        Log.i(TAG, "Astronaut selected item determined: " + astronautSelectedItem + " (Confidence: " + localBestAstronautDetection.confidence + ")");

                        Mat imageWithAstronautDetection = imageForYolo5.clone();
                        List<ObjectDetector.Detection> singleDetectionList = Arrays.asList(localBestAstronautDetection);
                        yoloDetector.drawDetections(imageWithAstronautDetection, singleDetectionList);
                        api.saveMatImage(imageWithAstronautDetection, "nav_cam_image_astronaut_best_detection.png");
                        imageWithAstronautDetection.release();
                    } else {
                        Log.w(TAG, "Astronaut Recognition: No primary target (diamond, emerald, crystal) detected, or other items not considered.");
                    }
                } else {
                    Log.w(TAG, "Astronaut Recognition: No items detected in the (modified) astronaut image.");
                }
                imageForYolo5.release();
            }
            if (targetImage != null) targetImage.release();
        }

        int reqArea = 1;
        if (astronautSelectedItem != null) {
            for (int i = 0; i < targetItemsInArea.length; i++) {
                if (astronautSelectedItem.equals(targetItemsInArea[i])) {
                    reqArea = i + 1;
                    Log.i(TAG, "Astronaut's selected item '" + astronautSelectedItem + "' matches item from Area " + reqArea);
                    break;
                }
            }
            if (reqArea == 1) {
                Log.i(TAG, "Astronaut's selected item '" + astronautSelectedItem + "' does not match any of the special items in Areas 1-4: " + Arrays.toString(targetItemsInArea));
            }
        } else {
            Log.w(TAG, "Cannot determine required area because no specific item was identified for the astronaut.");
        }
        moveTo(finalLocations[reqArea-1]);
        api.saveMatImage(api.getMatNavCam(), "nav_cam_image_area1_final.png");
        api.notifyRecognitionItem(); // Indicates failure to recognize a *specific* item or no primary target chosen
        api.takeTargetItemSnapshot(); // Takes snapshot of current view
    }

    @Override
    protected void runPlan2(){
        // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }

    private void restrict() {
        for (int i = 0; i < 4; i++) {
            Point position = locations[i].position;
            Quaternion orientation = locations[i].orientation;
            double x = position.getX();
            double y = position.getY();
            double z = position.getZ();

            if (x < 10.25) x = 10.25;
            if (x > 11.50) x = 11.50;
            if (y < -10.15) y = -10.15;
            if (y > -5.95) y = -5.95;
            if (z < 4.27) z = 4.27;
            if (z > 5.52) z = 5.52;

            locations[i] = new WorldPose(new Point(x, y, z), orientation);
        }
    }

    private Mat extractSymbolFromCam(boolean isAstronaut,int targetId) {
        Mat image = api.getMatNavCam();
        for(int i = 0; i < 3 && image.empty(); i++) {
            Log.i(TAG, "Retrying to get image from NavCam, attempt " + (i + 1));
            image = api.getMatNavCam();
        }
        if (image.empty()) {
            Log.e(TAG, "Failed to get image from NavCam for symbol extraction.");
            return null;
        }
        api.saveMatImage(image, "nav_cam_original_image" + targetId + ".png");
        if(targetId > 100)
            findFinalLocation(image, targetId);
        Mat modifiedImage = symbolExtractor.extractSymbol(image, targetId, isAstronaut);
        if (modifiedImage.empty()) {
            Log.e(TAG, "Failed to extract symbol from NavCam image.");
            return null;
        }
        api.saveMatImage(modifiedImage, "nav_cam_image" + targetId + ".png");
        return modifiedImage;
    }

    private void moveTo(WorldPose location) {
        for(int i = 0;i < 3;i++) {
            if(api.moveTo(location.position,location.orientation,false).hasSucceeded())
                return;
        }
    }

    private void intrinsicsToMat(double[][] intrinsics,boolean isSimulation) {
        double[] cameraMatrixData = new double[9];
        double[] distCoeffsData = new double[5];
        if (intrinsics == null || intrinsics.length < 2 || intrinsics[0].length != 9 || intrinsics[1].length < 5) {
            Log.e("FlattenImage", "Invalid camera intrinsics received. Using default.");
            if(isSimulation) {
                cameraMatrixData = new double[]{
                    523.105750, 0.000000, 635.434258,
                    0.000000, 534.765913, 500.335102,
                    0.000000, 0.000000, 1.000000
                };
                distCoeffsData = new double[]{-0.164787, 0.020375, -0.001572, -0.000369, 0.000000};
            } else {
                cameraMatrixData = new double[]{
                    608.8073, 0.0, 632.53684,
                    0.0, 607.61439, 549.08386,
                    0.0, 0.0, 1.0
                };
                distCoeffsData = new double[]{-0.212191, 0.073843, -0.000918, 0.001890, 0.0};
            }
        } else {
            System.arraycopy(intrinsics[0], 0, cameraMatrixData, 0, 9);
            System.arraycopy(intrinsics[1], 0, distCoeffsData, 0, 5);
        }
        // Use local variables for initialization
        cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraMatrix.put(0, 0, cameraMatrixData);
        distCoeffs = new Mat(1, 5, CvType.CV_64F);
        distCoeffs.put(0, 0, distCoeffsData);
    }

    private void findFinalLocation(Mat image, int targetId) {
        if (image == null || image.empty()) {
            Log.e("ARucoPose", "Input image is null or empty.");
            return;
        }
        int areaId = targetId - 100;
        double[] distance = estimateArucoPose(image, targetId);;
        Point currentPosition = locations[areaId-1].position;
        Quaternion currentOrientation = locations[areaId-1].orientation;
        if(areaId == 1) {
            double x = currentPosition.getX() + distance[0];
            double y = currentPosition.getY() - distance[2] + 0.75;
            double z = currentPosition.getZ() + distance[1];
            finalLocations[0] = new WorldPose(new Point(x, y, z), currentOrientation);
            Log.i("ARucoPose", String.format("Final Location for Area 1: X=%.3f m, Y=%.3f m, Z=%.3f m",
                    x, y, z));
        }
        if(areaId == 2 || areaId == 3) {
            double x = currentPosition.getX() + distance[0];
            double y = currentPosition.getY() - distance[1];
            double z = currentPosition.getZ() - distance[2] + 0.75;
            finalLocations[areaId-1] = new WorldPose(new Point(x, y, z), currentOrientation);
            Log.i("ARucoPose", String.format("Final Location for Area 2/3: X=%.3f m, Y=%.3f m, Z=%.3f m",
                    x, y, z));
        }
        if(areaId == 4) {
            double x = currentPosition.getX() - distance[2] + 0.75;
            double y = currentPosition.getY() + distance[0];
            double z = currentPosition.getZ() + distance[1];
            finalLocations[3] = new WorldPose(new Point(x, y, z), currentOrientation);
            Log.i("ARucoPose", String.format("Final Location for Area 4: X=%.3f m, Y=%.3f m, Z=%.3f m",
                    x, y, z));
        }
    }

    private static double[] estimateArucoPose(Mat image, int targetId) {
        Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        DetectorParameters parameters = DetectorParameters.create();

        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();

        // Detect markers
        Aruco.detectMarkers(image, arucoDict, corners, ids, parameters);
        Log.i("ARucoPose", "Detected " + ids.rows() + " ArUco markers.");
        if (!ids.empty()) {
            float markerLength = 0.05f;  // Marker size in meters (5 cm)
            Mat rvecs = new Mat();
            Mat tvecs = new Mat();

            Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

            for (int i = 0; i < ids.rows(); i++) {
                int markerId = (int) ids.get(i, 0)[0];

                if (markerId == targetId) {
                    double[] tvec = tvecs.row(i).get(0, 0);  // 1x3 vector
                    Log.e("ARucoPose", String.format("Marker ID %d 3D Position (Camera Coordinates): X=%.3f m, Y=%.3f m, Z=%.3f m",
                            markerId, tvec[0], tvec[1], tvec[2]));
                    return tvec;
                }
            }

            Log.e("ARucoPose","Target ArUco ID not found in image.");
        } else {
            Log.e("ARucoPose","No ArUco markers detected.");
        }
        return new double[]{0.0, 0.0, 0.0};
    }
}