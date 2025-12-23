using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class RayPerceptionBaseSensor3DManager : BaseSensorManager
{
    public GameObject rayContainingParentGameobject;
    private RayPerceptionSensorComponent3D[] m_RayPerceptionSensorComponent3Ds;
    
    public float rayLength = 10;

    
    public override void Initialize(SensorManagerProperties properties)
    {
        observationStacks = properties.ObservationStacks;
        SetProperties();
    }

    protected override void InitializeSensor()
    {
        if (!rayContainingParentGameobject) rayContainingParentGameobject = gameObject;
        m_RayPerceptionSensorComponent3Ds = rayContainingParentGameobject.GetComponentsInChildren<RayPerceptionSensorComponent3D>();
        environmentParameters = Academy.Instance.EnvironmentParameters;
        
        if (environmentParameters != null)
        {
            StoreProperties();
        }
        
        if (m_RayPerceptionSensorComponent3Ds.Length>0)
        {
            if (rayLength <= 0 || !useSensor)
            {
                rayContainingParentGameobject.SetActive(false);
            }
            else
            {
                rayContainingParentGameobject.SetActive(true);
            }
        }
    }

    protected override void StoreProperties()
    {
        useSensor = environmentParameters.GetWithDefault("use_raycasts", useSensor?1:0)>0;
        rayLength = environmentParameters.GetWithDefault("ray_length", rayLength);
        observationStacks= (int) environmentParameters.GetWithDefault("observation_stacks", observationStacks);
    }

    /// <summary>
    /// Sets the properties of the CameraSensor from values
    /// </summary>
    public override void SetProperties()
    {
        InitializeSensor();

        if (m_RayPerceptionSensorComponent3Ds.Length==0)
        {
            Debug.LogWarning("No RayPerceptionSensorComponent3Ds found");;
            return;
        }

        foreach (var sensor in m_RayPerceptionSensorComponent3Ds)
        {
            sensor.RayLength = rayLength;
            sensor.ObservationStacks = observationStacks;
            sensor.CreateSensors();
        }
    }
}