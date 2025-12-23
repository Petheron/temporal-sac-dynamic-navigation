using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.SideChannels;
using System;
using Unity.MLAgents;

public class TrajectoryPositionsSideChannel:SideChannel
{
    public TrajectoryPositionsSideChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        throw new UnityAgentsException("StatsSideChannel should never receive messages.");
    }

    public void AddTrajectoryPositions(int episode_id, List<Vector3> positions, EndCondition endCondition, Vector3 goalPosition)
    {
        var splitPositions = new List<float>();
        foreach(var pos in positions)
        {
            for (int i = 0; i < 3; i++)
            {
                splitPositions.Add(pos[i]);
            }
        }

        var goalPositionList = new List<float> {goalPosition.x, goalPosition.y, goalPosition.z};
        using (var msg = new OutgoingMessage())
        {
            msg.WriteInt32(episode_id);
            msg.WriteFloatList(splitPositions);
            msg.WriteString(endCondition.ToString());
            msg.WriteFloatList(goalPositionList);
            QueueMessageToSend(msg);
        }
    }
}
