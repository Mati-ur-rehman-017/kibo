package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.util.Log;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.core.Scalar; // Added missing import
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SymbolExtractor {
    private static final String TAG = "SymbolExtractor";
    // Changed from static final to instance final variables
    private final Mat cameraMatrix;
    private final Mat distCoeffs;
    private static final boolean isSimulation = true;

    // Added a private constructor to be called by the factory method
    private SymbolExtractor(Mat cameraMatrix, Mat distCoeffs) {
        this.cameraMatrix = cameraMatrix;
        this.distCoeffs = distCoeffs;
    }

    public static SymbolExtractor create(double[][] intrinsics) {
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
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraMatrix.put(0, 0, cameraMatrixData);
        Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
        distCoeffs.put(0, 0, distCoeffsData);
        // Call the new constructor
        return new SymbolExtractor(cameraMatrix, distCoeffs);
    }

    private Mat flattenImage(Mat image) {
        Mat flattenedImage = new Mat();
        Calib3d.undistort(image, flattenedImage, cameraMatrix, distCoeffs);
        if(flattenedImage.empty()) {
            Log.e(TAG, "Failed to flatten the image. Returning original.");
            return image; // Return original if undistortion fails
        }
        Log.d(TAG, "Image flattened successfully.");
        return flattenedImage;
    }
    /**
     * Crops the image based on the corners of a trapezoid defined by the ARuco marker.
     * The corners are expected to be in the order: top-left, top-right, bottom-right, bottom-left.
     *
     * @param image   The source image in Mat format.
     * @param corners A 2D array of corners defining the trapezoid.
     * @return A new Mat containing the cropped image.
     */
    private Mat cropImage(Mat image, float[][] corners) {
        if (corners == null || corners.length != 4) {
            Log.e(TAG, "Invalid corners for cropping.");
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
        if(resized.empty()) {
            Log.e(TAG, "Failed to resize the cropped image.");
            return image; // Return original if resizing fails
        }
        return resized;
    }

    private Mat unwrapImage(Mat image,int targetID) {
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
            if ((int) ids.get(i, 0)[0] == targetID) {
                markerIndex = i;
                break;
            }
        }

        if (markerIndex == -1) {
            Log.e("YourService", "Target marker with ID " + targetID + " not found for straightening.");
            ids.release(); // Release the empty Mat
            for (Mat cornerMat : corners) {
                cornerMat.release();
            }
            Log.e("YourService", "Returning original image.");
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
        if(straightenedImage.empty()) {
            Log.e(TAG, "Failed to straighten the image. Returning original.");
            return image; // Return original if warping fails
        }
        return straightenedImage;
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
        int cropX = (int) (minX - (5 * arucoPixelWidth));
        int cropY = (int) (minY - (0.5 * arucoPixelWidth));
        int cropRight = (int) (maxX + (-1 * arucoPixelWidth));
        int cropBottom = (int) (maxY + (2.5 * arucoPixelWidth));

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
        if(croppedImage.empty()) {
            Log.e("ArucoCrop", "Failed to crop the image. Returning original image.");
            return inputImage; // Return original if cropping fails
        }
        return croppedImage;
    }
    
    public Mat extractSymbol(Mat inputImage,int targetID,boolean isAstronaut) {
        Mat flatImage = flattenImage(inputImage);
        if (flatImage.empty()) {
            Log.e("YourService", "Failed to flatten the image.");
            return null;
        }
        Mat unwrappedImage = flatImage;
        if(!isAstronaut) {
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
                int[] idData = new int[(int) ids.total()];
                ids.get(0, 0, idData);

                int markerIndex = -1;
                for (int i = 0; i < idData.length; i++) {
                    if (idData[i] == targetID) {
                        markerIndex = i;
                        Log.i("YourService", "Found target marker with ID " + targetID + " at index " + i);
                        break; // Exit the loop once we've found our marker
                    }
                }

                if (markerIndex == -1) {
                    // We looped through all detected markers but didn't find the one we wanted.
                    Log.e("YourService", "Target marker with ID " + targetID + " was not found in the image.");
                    return null;
                }

                // Now, use the correct index to get the corners for our target marker
                Mat markerMat = corners.get(markerIndex);
                markerCorners = new float[4][2];

                // The corner Mat is of type CV_32F (float) and has a shape of 1x4 (with 2 channels)
                // or 4x1 (with 2 channels). Reading it into a flat float array is reliable.
                float[] cornerData = new float[(int) (markerMat.total() * markerMat.channels())];
                markerMat.get(0, 0, cornerData);

                for (int i = 0; i < 4; i++) {
                    markerCorners[i][0] = cornerData[i * 2];     // x
                    markerCorners[i][1] = cornerData[i * 2 + 1]; // y
                }
            }
            Mat cropImage = cropImage(flatImage, markerCorners);
            if (cropImage.empty()) {
                Log.e("YourService", "Failed to crop the image.");
                return null; // Fail quietly
            }
            unwrappedImage = unwrapImage(cropImage,targetID);
            if (unwrappedImage.empty()) {
                Log.e("YourService", "Failed to unwrap the image.");
                return null; // Fail quietly
            }
            org.opencv.imgproc.Imgproc.resize(unwrappedImage, unwrappedImage, new org.opencv.core.Size(800, unwrappedImage.rows() * 800.0 / unwrappedImage.cols()));
        }
        Log.i("YourService", "Image successfully processed and unwrapped.");
        return cropAroundArucoById(unwrappedImage, targetID);
    }
}