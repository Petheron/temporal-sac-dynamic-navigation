using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.SideChannels;
using System;

public class ActionLimitSideChannel : SideChannel
{
    public ActionLimitSideChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        var receivedMessage = msg.ReadString();
        Debug.Log("From Python: " + receivedMessage);
    }

    public void SendFloatListToPython(List<float> actionMinMaxes)
    {
        using var msgOut = new OutgoingMessage();
        msgOut.WriteFloatList(actionMinMaxes);
        QueueMessageToSend(msgOut);
        Debug.Log("Message to python: " + msgOut.ToString());
    }
}
