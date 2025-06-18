package jp.jaxa.iss.kibo.rpc.sampleapk;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

public class ArUcoLocalization {

    /**
     * Estimates the world location and orientation of the first detected ArUco tag.
     *
     * @param cameraWorldLocation The camera's current position in the world frame.
     * @param cameraWorldOrientation The camera's current orientation in the world frame.
     * @param inputImage The image captured by the camera.
     * @param cameraMatrix The camera's intrinsic matrix.
     * @param distCoeffs The camera's distortion coefficients.
     * @param markerSizeInMeters The physical size of the ArUco tag's side in meters.
     * @return A WorldPose object for the tag, or null if no tag is found.
     */
    public static WorldPose estimateTagWorldPose(
            Point cameraWorldLocation,
            Quaternion cameraWorldOrientation,
            Mat inputImage,
            Mat cameraMatrix,
            MatOfDouble distCoeffs,
            float markerSizeInMeters) {

        // 1. Detect ArUco markers in the image
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_4X4_50);
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        Aruco.detectMarkers(inputImage, dictionary, corners, ids);

        if (ids.total() == 0) {
            System.out.println("No ArUco tags found in the image.");
            return null; // No markers found
        }

        // 2. Estimate the pose of the marker *relative to the camera*
        Mat rvecs = new Mat(); // Rotation vectors
        Mat tvecs = new Mat(); // Translation vectors
        Aruco.estimatePoseSingleMarkers(corners, markerSizeInMeters, cameraMatrix, distCoeffs, rvecs, tvecs);

        // We'll use the first detected marker
        Mat rvec_cam_tag = new Mat();
        rvecs.row(0).copyTo(rvec_cam_tag);
        Mat tvec_cam_tag = new Mat();
        tvecs.row(0).copyTo(tvec_cam_tag);

        // 3. Convert poses into 4x4 transformation matrices for calculations
        // Matrix for the Tag's pose relative to the Camera
        Mat T_cam_tag = poseToMatrix(rvec_cam_tag, tvec_cam_tag);

        // Matrix for the Camera's pose relative to the World
        Mat T_world_cam = poseToMatrix(cameraWorldOrientation, cameraWorldLocation);

        // 4. Calculate the Tag's pose in the World frame
        // Formula: T_world_tag = T_world_cam * T_cam_tag
        Mat T_world_tag = new Mat();
        Core.gemm(T_world_cam, T_cam_tag, 1, new Mat(), 0, T_world_tag);

        // 5. Decompose the final world matrix back into a Point and Quaternion
        return matrixToPose(T_world_tag);
    }
    
    /**
     * Calculates a desired camera pose to be a certain distance in front of a tag.
     *
     * @param tagWorldPose The absolute pose of the target tag.
     * @param desiredDistance The desired distance from the tag (e.g., 0.5 meters).
     * @return A WorldPose object representing the desired location for the camera.
     */
    public static WorldPose calculateTargetCameraPose(WorldPose tagWorldPose, double desiredDistance) {
        // 1. Get the tag's pose as a 4x4 matrix
        Mat T_world_tag = poseToMatrix(tagWorldPose.orientation, tagWorldPose.position);

        // 2. Define the desired camera pose *relative to the tag*
        //    - Position: `desiredDistance` along the tag's positive Z-axis.
        //      (Standard ArUco convention has Z pointing OUT of the tag)
        //    - Orientation: Rotated 180 degrees around the Y-axis so the camera's -Z faces the tag's +Z.
        Mat T_tag_camera_desired = new Mat(4, 4, org.opencv.core.CvType.CV_64F, new Scalar(0));
        T_tag_camera_desired.put(0, 0, -1.0); // Rotation X
        T_tag_camera_desired.put(1, 1, 1.0);  // Rotation Y
        T_tag_camera_desired.put(2, 2, -1.0); // Rotation Z
        T_tag_camera_desired.put(2, 3, desiredDistance); // Translation Z
        T_tag_camera_desired.put(3, 3, 1.0);

        // 3. Calculate the desired camera pose in the World frame
        // Formula: T_world_camera_desired = T_world_tag * T_tag_camera_desired
        Mat T_world_camera_desired = new Mat();
        Core.gemm(T_world_tag, T_tag_camera_desired, 1, new Mat(), 0, T_world_camera_desired);

        // 4. Decompose the final matrix back to a Point and Quaternion
        return matrixToPose(T_world_camera_desired);
    }

    // ... (Inside the ArUcoLocalization class)

    // Helper to convert OpenCV rvec/tvec to a 4x4 transformation matrix
    private static Mat poseToMatrix(Mat rvec, Mat tvec) {
        Mat R = new Mat();
        Calib3d.Rodrigues(rvec, R); // Convert rotation vector to 3x3 rotation matrix
        Mat T = new Mat(4, 4, R.type());
        T.put(0, 0, R.get(0, 0)[0], R.get(0, 1)[0], R.get(0, 2)[0], tvec.get(0, 0)[0]);
        T.put(1, 0, R.get(1, 0)[0], R.get(1, 1)[0], R.get(1, 2)[0], tvec.get(1, 0)[0]);
        T.put(2, 0, R.get(2, 0)[0], R.get(2, 1)[0], R.get(2, 2)[0], tvec.get(2, 0)[0]);
        T.put(3, 0, 0, 0, 0, 1);
        return T;
    }

    // Helper to convert Astrobee Quaternion/Point to a 4x4 transformation matrix
    private static Mat poseToMatrix(Quaternion q, Point p) {
        Mat R = quaternionToRotationMatrix(q);
        Mat T = new Mat(4, 4, R.type());
        T.put(0, 0, R.get(0, 0)[0], R.get(0, 1)[0], R.get(0, 2)[0], p.getX());
        T.put(1, 0, R.get(1, 0)[0], R.get(1, 1)[0], R.get(1, 2)[0], p.getY());
        T.put(2, 0, R.get(2, 0)[0], R.get(2, 1)[0], R.get(2, 2)[0], p.getZ());
        T.put(3, 0, 0, 0, 0, 1);
        return T;
    }

    // Helper to convert a 4x4 transformation matrix back to a WorldPose
    private static WorldPose matrixToPose(Mat T) {
        // Extract position
        Point position = new Point(T.get(0, 3)[0], T.get(1, 3)[0], T.get(2, 3)[0]);

        // Extract rotation matrix
        Mat R = T.submat(0, 3, 0, 3);
        Quaternion orientation = rotationMatrixToQuaternion(R);

        return new WorldPose(position, orientation);
    }

    // Math helper: Convert Astrobee Quaternion to 3x3 OpenCV Rotation Matrix
    private static Mat quaternionToRotationMatrix(Quaternion q) {
        Mat R = new Mat(3, 3, org.opencv.core.CvType.CV_64F);
        double x = q.getX(), y = q.getY(), z = q.getZ(), w = q.getW();
        R.put(0, 0, 1 - 2 * y * y - 2 * z * z);
        R.put(0, 1, 2 * x * y - 2 * z * w);
        R.put(0, 2, 2 * x * z + 2 * y * w);
        R.put(1, 0, 2 * x * y + 2 * z * w);
        R.put(1, 1, 1 - 2 * x * x - 2 * z * z);
        R.put(1, 2, 2 * y * z - 2 * x * w);
        R.put(2, 0, 2 * x * z - 2 * y * w);
        R.put(2, 1, 2 * y * z + 2 * x * w);
        R.put(2, 2, 1 - 2 * x * x - 2 * y * y);
        return R;
    }

    // Math helper: Convert 3x3 OpenCV Rotation Matrix to Astrobee Quaternion
    private static Quaternion rotationMatrixToQuaternion(Mat R) {
        double trace = R.get(0, 0)[0] + R.get(1, 1)[0] + R.get(2, 2)[0];
        double w, x, y, z;
        if (trace > 0) {
            double S = Math.sqrt(trace + 1.0) * 2;
            w = 0.25 * S;
            x = (R.get(2, 1)[0] - R.get(1, 2)[0]) / S;
            y = (R.get(0, 2)[0] - R.get(2, 0)[0]) / S;
            z = (R.get(1, 0)[0] - R.get(0, 1)[0]) / S;
        } else if ((R.get(0, 0)[0] > R.get(1, 1)[0]) & (R.get(0, 0)[0] > R.get(2, 2)[0])) {
            double S = Math.sqrt(1.0 + R.get(0, 0)[0] - R.get(1, 1)[0] - R.get(2, 2)[0]) * 2;
            w = (R.get(2, 1)[0] - R.get(1, 2)[0]) / S;
            x = 0.25 * S;
            y = (R.get(0, 1)[0] + R.get(1, 0)[0]) / S;
            z = (R.get(0, 2)[0] + R.get(2, 0)[0]) / S;
        } else if (R.get(1, 1)[0] > R.get(2, 2)[0]) {
            double S = Math.sqrt(1.0 + R.get(1, 1)[0] - R.get(0, 0)[0] - R.get(2, 2)[0]) * 2;
            w = (R.get(0, 2)[0] - R.get(2, 0)[0]) / S;
            x = (R.get(0, 1)[0] + R.get(1, 0)[0]) / S;
            y = 0.25 * S;
            z = (R.get(1, 2)[0] + R.get(2, 1)[0]) / S;
        } else {
            double S = Math.sqrt(1.0 + R.get(2, 2)[0] - R.get(0, 0)[0] - R.get(1, 1)[0]) * 2;
            w = (R.get(1, 0)[0] - R.get(0, 1)[0]) / S;
            x = (R.get(0, 2)[0] + R.get(2, 0)[0]) / S;
            y = (R.get(1, 2)[0] + R.get(2, 1)[0]) / S;
            z = 0.25 * S;
        }
        return new Quaternion((float)x, (float)y, (float)z, (float)w);
    }
}