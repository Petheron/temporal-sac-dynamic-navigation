using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class TrajectoryPositionsRecorder
{
    private TrajectoryPositionsSideChannel m_TrajectoryPositionsSideChannel;

    public TrajectoryPositionsRecorder()
    {
        m_TrajectoryPositionsSideChannel = new TrajectoryPositionsSideChannel();
        SideChannelManager.RegisterSideChannel(m_TrajectoryPositionsSideChannel);
    }

    public void Add(int episodeID, List<Vector3> positions, EndCondition endCondition, Vector3 goalPosition)
    {
        m_TrajectoryPositionsSideChannel.AddTrajectoryPositions(episodeID, positions, endCondition, goalPosition);
    }

    private void Dispose()
    {
        SideChannelManager.UnregisterSideChannel(m_TrajectoryPositionsSideChannel);
    }
}
