using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActionLimits : MonoBehaviour
{
    public List<float> actionMins;
    private List<float> actionMaxs;

    public void SetActionMinMaxs(List<float> mins, List<float> maxs)
    {
        actionMins = mins;
        actionMaxs = maxs;
    }

    public List<float> GetActionMaxs()
    {
        return actionMaxs;
    }

    public List<float> GetActionMins()
    {
        return actionMins;
    }
}
