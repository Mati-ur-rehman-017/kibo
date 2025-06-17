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
        // --- Area 1 ---
        Point point1 = new Point(10.9922d, -9.4623d, 5.2776d);
        Quaternion quat1 = new Quaternion(0f, 0f, -0.7071f, 0.7071f);
        api.moveTo(point1, quat1, false);
        Kinematics pos_now = api.getRobotKinematics();
        Log.i("Confident", String.valueOf(pos_now.getConfidence()));
        Log.i("Position", String.valueOf(pos_now.getPosition()));
        Mat image1 = api.getMatNavCam();
        if(image1.empty()) {
            Log.e("YourService", "Failed to get image from NavCam at Area 1.");
        } else {
            api.saveMatImage(image1, "nav_cam_image_area1.png");
            api.saveMatImage(flattenImage(image1), "nav_cam_image_area1_flattened.png");
            Mat modifiedImage = extractSymbol(image1);
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
                api.setAreaInfo(1,det.className,bb);
                bb=bb+1;
            }
            // api.setAreaInfo(1, "item_name", 1);
        }

        // --- Area 2 ---
        Point point2 = new Point(11.0106d, -8.8328d, 4.87973d);
        Quaternion quat2 = new Quaternion(0f, -0.707f, 0f, 0.707f);
        api.moveTo(point2, quat2, false);
        pos_now = api.getRobotKinematics();
        Log.i("Confident", String.valueOf(pos_now.getConfidence()));
        Log.i("Position", String.valueOf(pos_now.getPosition()));
        Mat image2 = api.getMatNavCam();
        if(image2.empty()) {
            Log.e("YourService", "Failed to get image from NavCam at Area 2.");
        } else {
            api.saveMatImage(image2, "nav_cam_image_area2.png");
            // Mat modifiedImage2 = extractSymbol(image2);
            // api.saveMatImage(modifiedImage2, "nav_cam_image_area2_modified.png");
            // api.setAreaInfo(2, "item_name", 1);
        }

        // --- Area 3 ---
        Point point3 = new Point(11.0106d, -7.8828d, 4.87863d);
        Quaternion quat3 = new Quaternion(0f, -0.707f, 0f, 0.707f);
        api.moveTo(point3, quat3, false);
        pos_now = api.getRobotKinematics();
        Log.i("Confident", String.valueOf(pos_now.getConfidence()));
        Log.i("Position", String.valueOf(pos_now.getPosition()));
        Mat image3 = api.getMatNavCam();
        if(image3.empty()) {
            Log.e("YourService", "Failed to get image from NavCam at Area 3.");
        } else {
            api.saveMatImage(image3, "nav_cam_image_area3.png");
            // api.setAreaInfo(3, "item_name", 1);
        }

        // --- Area 4 ---
        Point point4 = new Point(10.984684d, -6.8947d, 5.0276d);
        Quaternion quat4 = new Quaternion(0f, 0f, 1f, 0f);
        api.moveTo(point4, quat4, false);
        pos_now = api.getRobotKinematics();
        Log.i("Confident", String.valueOf(pos_now.getConfidence()));
        Log.i("Position", String.valueOf(pos_now.getPosition()));
        Mat image4 = api.getMatNavCam();
        if(image4.empty()) {
            Log.e("YourService", "Failed to get image from NavCam at Area 4.");
        } else {
            api.saveMatImage(image4, "nav_cam_image_area4.png");
            // api.setAreaInfo(4, "item_name", 1);
        }

        // Continue with the rest of your mission
        api.reportRoundingCompletion();
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
    // You can add your method.
public Mat flattenImage(Mat distortedImage) {
    // 1. Get the camera intrinsic parameters
    double[][] intrinsics = api.getNavCamIntrinsics();

    // 2. Validate the retrieved parameters
    if (intrinsics == null || intrinsics.length < 2 || intrinsics[0].length != 9 || intrinsics[1].length < 4) {
        Log.e("FlattenImage", "Invalid camera intrinsics received. Returning original image.");
        return distortedImage;
    }

    // 3. Convert the primitive double arrays into OpenCV Mat objects
    double[] cameraMatrixData = intrinsics[0];
    double[] distCoeffsData = intrinsics[1];

    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
    cameraMatrix.put(0, 0, cameraMatrixData);

    Mat distCoeffs = new Mat(1, distCoeffsData.length, CvType.CV_64F);
    distCoeffs.put(0, 0, distCoeffsData);

    // 4. Perform the undistortion using the fisheye-specific function
    Mat flattenedImage = new Mat();

    // *** THIS IS THE CORRECTED LINE ***
    // Use the function from the Calib3d module designed for fisheye lenses.
    // It takes 4 arguments: src, dst, cameraMatrix (K), and distCoeffs (D).
    Calib3d.undistort(distortedImage, flattenedImage, cameraMatrix, distCoeffs);

    // 5. Release intermediate Mats to free up native memory
    cameraMatrix.release();
    distCoeffs.release();

    return flattenedImage;
}


    private Mat cropImage(Mat image, float[][] corners) {
        if (corners == null || corners.length != 4) {
            Log.e("YourService", "Invalid corners for cropping.");
            return image;
        }

        // Find the center of the trapezoid
        float centerX = 0, centerY = 0;
        for (int i = 0; i < 4; i++) {
            centerX += corners[i][0];
            centerY += corners[i][1];
        }
        centerX /= 4.0f;
        centerY /= 4.0f;

        // Find the longest side of the trapezoid
        double maxLen = 0;
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            double dx = corners[i][0] - corners[j][0];
            double dy = corners[i][1] - corners[j][1];
            double len = Math.sqrt(dx * dx + dy * dy);
            if (len > maxLen) maxLen = len;
        }
        int radius = (int) (8 * maxLen);

        // Create a white background image
        int size = radius * 2;
        Mat whiteBg = new Mat(size, size, image.type(), new Scalar(255, 255, 255));

        // Define the ROI in the source image
        org.opencv.core.Point center = new org.opencv.core.Point(centerX, centerY);

        // Compute the bounding box for the circle in the source image
        int x0 = (int) (centerX - radius);
        int y0 = (int) (centerY - radius);
        int x1 = (int) (centerX + radius);
        int y1 = (int) (centerY + radius);

        // Compute the region to copy from the source image
        int srcX0 = Math.max(x0, 0);
        int srcY0 = Math.max(y0, 0);
        int srcX1 = Math.min(x1, image.cols());
        int srcY1 = Math.min(y1, image.rows());

        // Compute the region in the destination image
        int dstX0 = srcX0 - x0;
        int dstY0 = srcY0 - y0;
        int dstX1 = dstX0 + (srcX1 - srcX0);
        int dstY1 = dstY0 + (srcY1 - srcY0);

        // Copy the region from the source image to the white background
        Mat srcRoi = image.submat(srcY0, srcY1, srcX0, srcX1);
        Mat dstRoi = whiteBg.submat(dstY0, dstY1, dstX0, dstX1);
        srcRoi.copyTo(dstRoi);

        Mat resized = new Mat();
        org.opencv.imgproc.Imgproc.resize(whiteBg, resized, new org.opencv.core.Size(whiteBg.cols() * 3, whiteBg.rows() * 3));
        return resized;
    }

private Mat unwrapImage(Mat image) {
    // 1. Detect ARuco markers
    Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
    DetectorParameters detectorParams = DetectorParameters.create();
    List<Mat> corners = new ArrayList<>();
    Mat ids = new Mat();
    Aruco.detectMarkers(image, arucoDict, corners, ids, detectorParams);

    if (ids.empty() || corners.isEmpty()) {
        Log.e("YourService", "No ARuco markers detected for straightening.");
        return image; // Return original if no markers found
    }

    // 2. Find marker with ID 101
    int markerIndex = -1;
    for (int i = 0; i < ids.rows(); i++) {
        if ((int) ids.get(i, 0)[0] == 101) {
            markerIndex = i;
            break;
        }
    }

    if (markerIndex == -1) {
        Log.e("YourService", "ARuco marker with ID 101 not found.");
        return image; // Return original if target marker not found
    }

    // Get the source points (the detected, distorted marker corners)
    Mat markerCornersMat = corners.get(markerIndex);
    // Using fully qualified name to avoid ambiguity with other Point classes
    org.opencv.core.Point[] srcPts = new org.opencv.core.Point[4];
    float[] data = new float[8];
    markerCornersMat.get(0, 0, data);
    srcPts[0] = new org.opencv.core.Point(data[0], data[1]); // Top-left
    srcPts[1] = new org.opencv.core.Point(data[2], data[3]); // Top-right
    srcPts[2] = new org.opencv.core.Point(data[4], data[5]); // Bottom-right
    srcPts[3] = new org.opencv.core.Point(data[6], data[7]); // Bottom-left

    // 3. Define a standard, straightened marker shape (destination points for the marker)
    double markerSideLength = 200; // A standard reference size in pixels
    org.opencv.core.Point[] markerStandardPts = new org.opencv.core.Point[]{
        new org.opencv.core.Point(0, 0),
        new org.opencv.core.Point(markerSideLength - 1, 0),
        new org.opencv.core.Point(markerSideLength - 1, markerSideLength - 1),
        new org.opencv.core.Point(0, markerSideLength - 1)
    };

    // 4. Get the transform that maps the detected marker to the standard shape
    Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(
        Converters.vector_Point2f_to_Mat(Arrays.asList(srcPts)),
        Converters.vector_Point2f_to_Mat(Arrays.asList(markerStandardPts))
    );

    // 5. Find where the corners of the ORIGINAL IMAGE will land after this transform
    List<org.opencv.core.Point> imageCorners = Arrays.asList(
        new org.opencv.core.Point(0, 0),
        new org.opencv.core.Point(image.cols() - 1, 0),
        new org.opencv.core.Point(image.cols() - 1, image.rows() - 1),
        new org.opencv.core.Point(0, image.rows() - 1)
    );
    Mat imageCornersMat = Converters.vector_Point2f_to_Mat(imageCorners);
    Mat transformedCornersMat = new Mat();
    Core.perspectiveTransform(imageCornersMat, transformedCornersMat, perspectiveMatrix);

    // 6. Find the bounding box of the transformed corners to determine the new canvas size
    float[] transformedCornersData = new float[8];
    transformedCornersMat.get(0, 0, transformedCornersData);
    double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
    double maxX = Double.MIN_VALUE, maxY = Double.MIN_VALUE;

    for (int i = 0; i < 4; i++) {
        minX = Math.min(minX, transformedCornersData[i * 2]);
        maxX = Math.max(maxX, transformedCornersData[i * 2]);
        minY = Math.min(minY, transformedCornersData[i * 2 + 1]);
        maxY = Math.max(maxY, transformedCornersData[i * 2 + 1]);
    }

    double newWidth = maxX - minX;
    double newHeight = maxY - minY;

    // 7. Create a translation matrix to shift the image back to the (0,0) origin
    Mat translationMatrix = Mat.eye(3, 3, CvType.CV_64F);
    translationMatrix.put(0, 2, -minX); // Shift horizontally by -minX
    translationMatrix.put(1, 2, -minY); // Shift vertically by -minY

    // 8. Combine the translation with the original perspective transform
    Mat finalTransformMatrix = new Mat();
    // Multiply the translation matrix by the perspective matrix
    Core.gemm(translationMatrix, perspectiveMatrix, 1, new Mat(), 0, finalTransformMatrix);

    // 9. Warp the original image using the final, corrected transformation matrix
    Mat straightenedImage = new Mat();
    Imgproc.warpPerspective(
        image,
        straightenedImage,
        finalTransformMatrix,
        new Size(newWidth, newHeight)
    );

    // Release intermediate matrices to free up memory
    ids.release();
    markerCornersMat.release();
    imageCornersMat.release();
    transformedCornersMat.release();
    perspectiveMatrix.release();
    translationMatrix.release();
    finalTransformMatrix.release();


    return straightenedImage;
}

    private Mat extractSymbolFromNavCam() {
        Mat inputImage = api.getMatNavCam();
        if (inputImage.empty()) {
            Log.e("YourService", "Failed to get image from NavCam.");
            return null; // Fail quietly
        }
        return extractSymbol(inputImage);
    }
/**
 * Detects a specific ARuco marker by its ID and crops the image based on offsets
 * relative to the marker's size and position.
 *
 * @param inputImage The source image in Mat format.
 * @param targetId   The integer ID of the ARuco marker to crop around.
 * @return A new, cropped Mat. If the specified marker is not found or the crop is invalid,
 *         the original image Mat is returned.
 */
private Mat cropAroundArucoById(Mat inputImage, int targetId) {
    // 1. Setup for and perform ARuco marker detection
    Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
    DetectorParameters detectorParams = DetectorParameters.create();
    List<Mat> corners = new ArrayList<>();
    Mat ids = new Mat();
    Aruco.detectMarkers(inputImage, arucoDict, corners, ids, detectorParams);

    // 2. Check if any markers were found at all
    if (ids.empty() || corners.isEmpty()) {
        Log.e("ArucoCrop", "No ARuco markers detected at all. Returning original image.");
        ids.release(); // Release the empty Mat
        return inputImage;
    }

    // 3. Find the index of the marker with the specified targetId
    int markerIndex = -1;
    for (int i = 0; i < ids.rows(); i++) {
        if ((int) ids.get(i, 0)[0] == targetId) {
            markerIndex = i;
            break; // Found the marker, no need to search further
        }
    }

    // 4. Check if the specific marker was found
    if (markerIndex == -1) {
        Log.e("ArucoCrop", "Marker with ID " + targetId + " not found. Returning original image.");
        // Release all allocated Mats before returning
        ids.release();
        for (Mat cornerMat : corners) {
            cornerMat.release();
        }
        return inputImage;
    }

    // 5. Get the properties of the target marker
    Mat targetMarkerCorners = corners.get(markerIndex);
    float[] cornerData = new float[8];
    targetMarkerCorners.get(0, 0, cornerData);

    org.opencv.core.Point p0 = new org.opencv.core.Point(cornerData[0], cornerData[1]); // Top-left
    org.opencv.core.Point p1 = new org.opencv.core.Point(cornerData[2], cornerData[3]); // Top-right
    org.opencv.core.Point p2 = new org.opencv.core.Point(cornerData[4], cornerData[5]); // Bottom-right
    org.opencv.core.Point p3 = new org.opencv.core.Point(cornerData[6], cornerData[7]); // Bottom-left

    double minX = Math.min(p0.x, Math.min(p1.x, Math.min(p2.x, p3.x)));
    double maxX = Math.max(p0.x, Math.max(p1.x, Math.max(p2.x, p3.x)));
    double minY = Math.min(p0.y, Math.min(p1.y, Math.min(p2.y, p3.y)));
    double maxY = Math.max(p0.y, Math.max(p1.y, Math.max(p2.y, p3.y)));

    double topWidth = Math.sqrt(Math.pow(p1.x - p0.x, 2) + Math.pow(p1.y - p0.y, 2));
    double bottomWidth = Math.sqrt(Math.pow(p2.x - p3.x, 2) + Math.pow(p2.y - p3.y, 2));
    double arucoPixelWidth = (topWidth + bottomWidth) / 2.0;

    // 6. Calculate the desired crop boundaries based on the rules
    int cropX = (int) (minX - (6 * arucoPixelWidth));
    int cropY = (int) (minY - (1 * arucoPixelWidth));
    int cropRight = (int) (maxX + (1 * arucoPixelWidth));
    int cropBottom = (int) (maxY + (4 * arucoPixelWidth));

    int cropWidth = cropRight - cropX;
    int cropHeight = cropBottom - cropY;

    // 7. Safety clamp the crop rectangle to the image boundaries
    int finalCropX = Math.max(0, cropX);
    int finalCropY = Math.max(0, cropY);

    if (finalCropX + cropWidth > inputImage.cols()) {
        cropWidth = inputImage.cols() - finalCropX;
    }
    if (finalCropY + cropHeight > inputImage.rows()) {
        cropHeight = inputImage.rows() - finalCropY;
    }

    if (cropWidth <= 0 || cropHeight <= 0) {
        Log.e("ArucoCrop", "Calculated crop region is invalid for marker ID " + targetId + ". Returning original image.");
        // Release all allocated Mats before returning
        ids.release();
        for (Mat cornerMat : corners) {
            cornerMat.release();
        }
        return inputImage;
    }
    
    // 8. Create the cropping rectangle and perform the crop
    org.opencv.core.Rect cropRect = new org.opencv.core.Rect(finalCropX, finalCropY, cropWidth, cropHeight);
    Mat croppedImage = new Mat(inputImage, cropRect).clone();

    // 9. Release memory of all intermediate Mats
    ids.release();
    for (Mat cornerMat : corners) {
        cornerMat.release();
    }

    return croppedImage;
}
    private Mat extractSymbol(Mat inputImage) {
        Mat flatImage = flattenImage(inputImage);
        if (flatImage.empty()) {
            Log.e("YourService", "Failed to flatten the image.");
            return null; // Fail quietly
        }
        Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        DetectorParameters detectorParams = DetectorParameters.create();
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        float[][] markerCorners = new float[4][2];
        Aruco.detectMarkers(flatImage, arucoDict, corners, ids, detectorParams);
        if (ids.empty() || corners.isEmpty()) {
            Log.e("YourService", "No markers detected in the image.");
            return null; // Fail quietly
        }
        else {
            // Assuming corners.get(0) is a Mat of shape (4,1,2) or (4,2)
            Mat markerMat = corners.get(0);
            float[] data = new float[(int) (markerMat.total() * markerMat.channels())];
            markerMat.get(0, 0, data);
            for (int i = 0; i < 4; i++) {
                markerCorners[i][0] = data[i * 2];     // x
                markerCorners[i][1] = data[i * 2 + 1]; // y
            }
            Log.i("YourService", "Marker corners detected: " + markerCorners[0][0] + ", " + markerCorners[0][1] +
                    "; " + markerCorners[1][0] + ", " + markerCorners[1][1] +
                    "; " + markerCorners[2][0] + ", " + markerCorners[2][1] +
                    "; " + markerCorners[3][0] + ", " + markerCorners[3][1]);
        }
        Mat cropImage = cropImage(flatImage, markerCorners);
        if (cropImage.empty()) {
            Log.e("YourService", "Failed to crop the image.");
            return null; // Fail quietly
        }
        Mat unwrappedImage = unwrapImage(cropImage);
        if (unwrappedImage.empty()) {
            Log.e("YourService", "Failed to unwrap the image.");
            return null; // Fail quietly
        }
        org.opencv.imgproc.Imgproc.resize(unwrappedImage, unwrappedImage, new org.opencv.core.Size(800, unwrappedImage.rows() * 800.0 / unwrappedImage.cols()));
        Log.i("YourService", "Image successfully processed and unwrapped.");
        return cropAroundArucoById(unwrappedImage, 101);
    }
}
