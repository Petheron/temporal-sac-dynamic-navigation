using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;

public class RegisterActionLimitSideChannel : MonoBehaviour
{
    public bool registerActionLimits = false;

    ActionLimitSideChannel actionLimitChannel;

    public void Awake()
    {
        if (registerActionLimits)
        {
            actionLimitChannel = new ActionLimitSideChannel();

            ActionLimitsManager.OnSendActionLimits += actionLimitChannel.SendFloatListToPython;

            SideChannelManager.RegisterSideChannel(actionLimitChannel);
        }
    }

    public void OnDestroy()
    {
        if (registerActionLimits)
        {
            if (Academy.IsInitialized)
            {
                SideChannelManager.UnregisterSideChannel(actionLimitChannel);
            }

            ActionLimitsManager.OnSendActionLimits -= actionLimitChannel.SendFloatListToPython;
        }
        
    }
}
