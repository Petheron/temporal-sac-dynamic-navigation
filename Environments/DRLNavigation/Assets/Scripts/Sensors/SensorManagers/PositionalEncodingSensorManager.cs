using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public enum PositionalEncoderType
{
    Positional,
    Sestini
}

public class PositionalEncodingSensorManager : BaseSensorManager
{
    public VectorSensorComponent m_VectorSensorComponent;

    public String baseName = "Agent";

    public List<Transform> transformsToEncode;

    public bool useSestiniEncoding = false;

    private int numberOfPositionDimensions = 3;

    private int currentObsCount;

    public OccupancyMap occupancyMap;
    
    public override void Initialize(SensorManagerProperties properties)
    {
        observationStacks = properties.ObservationStacks;
        SetProperties();
    }

    protected override void InitializeSensor()
    {
        if (!m_VectorSensorComponent)
        {
            DestroyImmediate(this);
            return;
        }
        
        environmentParameters = Academy.Instance.EnvironmentParameters;
        
        if (environmentParameters != null)
        {
            StoreProperties();
        }

        if (numberOfPositionDimensions <= 0 || !useSensor)
        {
            DestroyImmediate(m_VectorSensorComponent);
            return;
        }

        if (useSestiniEncoding)
        {
            m_VectorSensorComponent.SensorName = baseName + "Sestini" + "EncodingSensor";
            Debug.Assert(m_VectorSensorComponent.SensorName.Contains("Sestini"));
        }
        else
        {
            m_VectorSensorComponent.SensorName = baseName + "Positional" + "EncodingSensor";
            Debug.Assert(m_VectorSensorComponent.SensorName.Contains("Positional"));
        }
         
        currentObsCount = 0;

        if (!occupancyMap) occupancyMap = FindObjectOfType<OccupancyMap>();
    }

    protected override void StoreProperties()
    {
        useSensor = environmentParameters.GetWithDefault("use_positional_embedding_sensor", useSensor?1:0) > 0.0f;
        useSestiniEncoding = environmentParameters.GetWithDefault("use_sestini_encoding", useSestiniEncoding ? 1.0f : 0f) > 0f;
        numberOfPositionDimensions = useSestiniEncoding ? 1 : 3;
        numberOfPositionDimensions *= transformsToEncode.Count;
        observationStacks= (int) environmentParameters.GetWithDefault("observation_stacks", observationStacks);
    }
    
    public override void SetProperties()
    {
        InitializeSensor();

        if (!m_VectorSensorComponent)
        {
            // Debug.LogWarning("VectorSensorComponent is null, cannot set observation size.");
            return;
        }

        m_VectorSensorComponent.ObservationSize = numberOfPositionDimensions;
        m_VectorSensorComponent.ObservationStacks = observationStacks;
        m_VectorSensorComponent.CreateSensors();
    }

    public void GetOccupancyMapPositionObservation()
    {
        if (!occupancyMap) return;
        if (transformsToEncode.Count==0) return;
        ResetObsCount();
        if (!useSestiniEncoding)
        {
            foreach (var t in transformsToEncode)
            {
                int[] dimIndex = occupancyMap.GetAllDimIndicesFromGlobalPosition(t.position);
                AddObservation(dimIndex);
            }
        }
        else
        {
            foreach (var t in transformsToEncode)
            {
                int[] index = occupancyMap.GetAllIndicesFromGlobalPosition(t.position);
                AddObservation(index);
            }
        }
        
    }

    public void ResetObsCount()
    {
        currentObsCount = 0;
    }
    
    
    public void AddObservation(Vector3Int obs)
    {
        if (currentObsCount + 3 > numberOfPositionDimensions || !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddObservation(obs);
        currentObsCount += 3;
    }
    
    public void AddObservation(int obs)
    {
        if (currentObsCount + 1 > numberOfPositionDimensions|| !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddObservation(obs);
        currentObsCount += 1;
    }
    
    public void AddObservation(Vector2Int obs)
    {
        if (currentObsCount + 2 > numberOfPositionDimensions|| !m_VectorSensorComponent)
        {
            return;
        }
        m_VectorSensorComponent.GetSensor().AddObservation(obs);
        currentObsCount += 2;
    }
    
    public void AddObservation(IList<int> obs)
    {
        if (currentObsCount + obs.Count > numberOfPositionDimensions|| !m_VectorSensorComponent)
        {
            return;
        }

        foreach (var o in obs)
        {
            m_VectorSensorComponent.GetSensor().AddObservation(o);
        }
        currentObsCount += obs.Count;
    }
    
}