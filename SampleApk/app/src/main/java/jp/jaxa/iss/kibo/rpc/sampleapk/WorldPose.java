package jp.jaxa.iss.kibo.rpc.sampleapk;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

// A simple container for a pose in the world frame.
public class WorldPose {
    public final Point position;
    public final Quaternion orientation;

    public WorldPose(Point position, Quaternion orientation) {
        this.position = position;
        this.orientation = orientation;
    }
}