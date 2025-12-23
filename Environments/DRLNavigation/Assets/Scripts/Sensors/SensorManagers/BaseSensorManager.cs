using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public abstract class BaseSensorManager:MonoBehaviour
{
    [ReadOnly] public int observationStacks = 1;
    public bool useSensor = true;
    protected EnvironmentParameters environmentParameters;

    public abstract void Initialize(SensorManagerProperties properties);
    protected abstract void InitializeSensor();
    protected abstract void StoreProperties();
    public abstract void SetProperties();
}
