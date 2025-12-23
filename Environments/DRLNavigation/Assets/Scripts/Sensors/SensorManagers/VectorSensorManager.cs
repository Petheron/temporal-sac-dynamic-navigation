using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class VectorSensorManager : BaseSensorManager
{
    private VectorSensorComponent m_VectorSensorComponent;

    public int observationSize = 8;

    private int currentObsCount;

    public List<float> debugVectorObservation;


    public override void Initialize(SensorManagerProperties properties)
    {
        observationStacks = properties.ObservationStacks;
        observationSize = properties.ObservationSize;
        SetProperties();
    }

    protected override void InitializeSensor()
    {
        m_VectorSensorComponent = GetComponent<VectorSensorComponent>();
        if (m_VectorSensorComponent == null)
        {
            m_VectorSensorComponent = GetComponentInChildren<VectorSensorComponent>();
        }
        environmentParameters = Academy.Instance.EnvironmentParameters;
        
        if (environmentParameters != null)
        {
            StoreProperties();
        }
        
        if (m_VectorSensorComponent)
        {
            if (observationSize <= 0 || !useSensor)
            {
                DestroyImmediate(m_VectorSensorComponent);
            }
        }

        currentObsCount = 0;
    }

    protected override void StoreProperties()
    {
        observationStacks= (int) environmentParameters.GetWithDefault("observation_stacks", observationStacks);
    }

    /// <summary>
    /// Sets the properties of the CameraSensor from values
    /// </summary>
    public override void SetProperties()
    {
        InitializeSensor();

        if (!m_VectorSensorComponent)
        {
            Debug.LogWarning("VectorSensorComponent is null, cannot set observation size.");;
            return;
        }

        m_VectorSensorComponent.ObservationSize = observationSize;
        m_VectorSensorComponent.ObservationStacks = observationStacks;
        m_VectorSensorComponent.CreateSensors();

        debugVectorObservation = new List<float>();
    }

    public void ResetObsCount()
    {
        currentObsCount = 0;
        debugVectorObservation.Clear();
    }

    public void AddObservation(float obs)
    {
        if (currentObsCount + 1 > observationSize|| !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddObservation(obs);
        currentObsCount += 1;
        
        debugVectorObservation.Add(obs);
    }
    
    public void AddObservation(Vector3 obs)
    {
        if (currentObsCount + 3 > observationSize || !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddObservation(obs);
        currentObsCount += 3;
        
        debugVectorObservation.AddRange(new List<float>{obs.x,obs.y,obs.z});
    }
    
    public void AddObservation(int obs)
    {
        if (currentObsCount + 1 > observationSize|| !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddObservation(obs);
        currentObsCount += 1;
        
        debugVectorObservation.Add(obs);
    }
    
    public void AddObservation(Vector2 obs)
    {
        if (currentObsCount + 2 > observationSize|| !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddObservation(obs);
        currentObsCount += 2;
        
        debugVectorObservation.AddRange(new List<float>{obs.x,obs.y});
    }
    
    public void AddObservation(IList<float> obs)
    {
        if (currentObsCount + obs.Count > observationSize|| !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddObservation(obs);
        currentObsCount += obs.Count;
        
        debugVectorObservation.AddRange(obs);
    }
    
    public void AddObservation(Quaternion obs)
    {
        if (currentObsCount + 4 > observationSize|| !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddObservation(obs);
        currentObsCount += 4;
        
        debugVectorObservation.AddRange(new List<float>{obs.x,obs.y, obs.z, obs.w});
    }
    
    public void AddObservation(bool obs)
    {
        if (currentObsCount + 1 > observationSize|| !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddObservation(obs);
        currentObsCount += 1;
        
        debugVectorObservation.Add(obs?1.0f:0f);
    }
    
    public void AddOneHotObservation(int obs, int range)
    {
        if (currentObsCount + range > observationSize|| !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddOneHotObservation(obs, range);
        currentObsCount += range;

        for (var i = 0; i < range; i++)
        {
            debugVectorObservation.Add(i==obs?1.0f:0f);
        }
    }
}