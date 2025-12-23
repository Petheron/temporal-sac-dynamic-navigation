using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActionLimitsManager : MonoBehaviour
{
/*    public float[] mins;
    public float[] maxs;*/

    public List<float> mins;
    public List<float> maxs;

    public delegate void SendActionLimits(List<float> minMax);

    public static event SendActionLimits OnSendActionLimits;


    public void SetActionLimits(ActionLimits actionLimits)
    {
        print("Setting action limits");
        actionLimits.SetActionMinMaxs(mins, maxs);
    }

    public void SendActions()
    {
        List<float> combinedMinMax = new List<float>();
        combinedMinMax.AddRange(mins);
        combinedMinMax.AddRange(maxs);
        print(combinedMinMax.ToString());
        OnSendActionLimits(combinedMinMax);
    }

    
}
