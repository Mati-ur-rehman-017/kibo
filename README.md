# Kibo Project Documentation

This project focuses on locating lost items using a combination of object detection, path optimization, and AR-guided navigation. The development was divided into three main stages:

1.  Data Set Preparation and Model Training
2.  Path Optimization and Model Deployment
3.  Logic for Navigational Return to a Found Item

## 1. Data Set Preparation and Model Training

The initial phase involved creating a robust dataset and training an efficient object detection model.

*   **Data Set Preparation:**
    *   Source images: "Lost item images" provided by Kibo.
    *   Preprocessing: Images were cropped. These cropped items were then randomly placed in new composite images.
    *   Augmentation considerations:
        *   Number of objects per image.
        *   Object overlapping.
        *   Varying DPI (Dots Per Inch) of the source images.
        *   Different orientations and sizes of objects.
    *   Creating such a dataset was only possible because of cropping images according to AR tag explained in step 2
*   **Model Selection:**
    *   Model: YOLOv5 Nano.
    *   Rationale: Chosen for its small model size, fast inference speed, and relatively accurate predictions suitable for deployment constraints.
*   **Model Training:**
    *   The model was trained using the official training script provided in the YOLOv5 Nano GitHub repository.
    *   The script used for training can be found in `train.py`.

## 2. Path Optimization and Model Deployment

This stage focused on deploying the trained model and developing strategies for effective item localization.

*   **Model Deployment:**
    *   The trained YOLOv5 Nano model was converted to the `.onnx` format.
    *   This `.onnx` model is loaded and utilized within the application for inference.
*   **Path Finding Strategy ("Oasis Zone"):**
    *   To effectively scan the "oasis zone," strategic observation points were determined.
    *   These points were selected based on repetitive observations to ensure they were numerous enough and at an adequate distance to capture clear, accurate images of the environment.
*   **Image Preprocessing for Enhanced Accuracy:**
    *   **AR Tag Utilization:** An AR tag was used to identify the specific target area (e.g., a page) within the camera's view.
    *   **Targeted Cropping:** A complex cropping mechanism was implemented to isolate the relevant page containing objects from the broader image.
    *   **Benefits:** This targeted approach significantly simplified the dataset creation process for training and improved the accuracy of object detection during runtime by focusing the model on the most relevant image region.

## 3. Logic for Navigational Return

The final stage implemented the logic for the robot/system to navigate back to the location of a previously identified item.

*   **Navigation Constraints:**
    *   Maintain a 30-degree angular constraint relative to the target.
    *   Maintain a 0.9 meters distance constraint from a reference.
*   **Navigation Method:**
    *   A straight vector is calculated towards the AR tag associated with the found item's location.
    *   The system moves to a position 0.75 meters away from this AR tag.
*   **Benefits:** This approach helped in minimizing angular error and ensured the system accurately pointed towards the correct image/item location.

## Key Files and Directories

*   **`transparent/`**: This folder contains the cropped images used for dataset preparation.
*   **`train.ipynb`**: This script was used for training the YOLOv5 Nano model.
*   **`kibo/SampleApk/app/src/main/java/jp/jaxa/iss/kibo/rpc/sampleapk/`**: The core application logic, including model loading, pathfinding, and navigation, is incorporated within this directory structure.

## Building the APK

To build the Android application (APK):

1.  Ensure your `ANDROID_HOME` environment variable is set correctly (e.g., `export ANDROID_HOME=$HOME/Android/Sdk`).
2.  Navigate to the `kibo/SampleApk/` directory (or the root directory containing `gradlew`).
3.  Run the following command:
    ```bash
    env ANDROID_HOME=$HOME/Android/Sdk ./gradlew assembleDebug
    ```
    *(Note: If you are already in a shell where `ANDROID_HOME` is set, you might just need `./gradlew assembleDebug`)*

4.  The resultant APK can be found in:
    `kibo/SampleApk/app/build/outputs/apk/debug/app-debug.apk`

---
