using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class TrajectoryPositionManager : MonoBehaviour
{
    [SerializeField] private bool storePositions;
    public TrajectoryPositionsRecorder TrajectoryPositionsRecorder { get; private set; }

    public void InitializeRecorder()
    {
        storePositions = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "store_positions",
            storePositions ? 1.0f : 0.0f) > 0;
        TrajectoryPositionsRecorder = storePositions ? new TrajectoryPositionsRecorder() : null;
    }
}
