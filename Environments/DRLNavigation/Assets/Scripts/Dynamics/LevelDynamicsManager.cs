using System;
using Unity.MLAgents;
using UnityEngine;

public class LevelDynamicsManager : MonoBehaviour
{
    public void SetLevelDynamics(bool enableDynamics)
    {
        foreach (var d in transform.GetComponentsInChildren<Dynamic>())
        {
            d.SetDynamicState(enableDynamics);
        }
    }
}
