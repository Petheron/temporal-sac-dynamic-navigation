using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class BoxcastSensorManager : BaseSensorManager
{
    public BoxcastSensorProperties boxcastSensorProperties;
    private BoxcastSensorComponent boxcastSensorComponent;
    
    [SerializeField] private int depth= 3;
    [SerializeField] private int height = 3;
    [SerializeField] private int width = 3;
    [SerializeField] private int numChannels = 1;

    public override void Initialize(SensorManagerProperties properties)
    {
        observationStacks = properties.ObservationStacks;
        SetProperties();
    }

    protected override void InitializeSensor()
    {
        boxcastSensorComponent = GetComponent<BoxcastSensorComponent>();
        if (!boxcastSensorComponent)
        {
            boxcastSensorComponent = GetComponentInChildren<BoxcastSensorComponent>();
        }
        environmentParameters = Academy.Instance.EnvironmentParameters;

        if (boxcastSensorProperties)
        {
            GetProperties();
        }
        
        if (environmentParameters != null)
        {
            StoreProperties();
        }

        if (boxcastSensorComponent)
        {
            if ((!useSensor || width <= 0 || height <= 0 || depth<=0))
            {
                DestroyImmediate(boxcastSensorComponent);
            }
        }

    }

    protected override void StoreProperties()
    {
        useSensor = environmentParameters.GetWithDefault("use_boxcast_sensor", useSensor?1:0) > 0.0f;
        width = (int) environmentParameters.GetWithDefault("bc_width", width);
        height = (int) environmentParameters.GetWithDefault("bc_height", height);
        depth = (int)environmentParameters.GetWithDefault("bc_depth", depth);
        numChannels= (int) environmentParameters.GetWithDefault("bc_num_channels", numChannels);
        
        observationStacks= (int) environmentParameters.GetWithDefault("observation_stacks", observationStacks);

        EnsureOddDims();
    }

    private void EnsureOddDims()
    {
        width = BaseGrid.EnsureOddDim(width);
        height = BaseGrid.EnsureOddDim(height);
        depth = BaseGrid.EnsureOddDim(depth);
    }

    public void SetDepthHeightWidth(int d, int h, int w)
    {
        depth = d;
        height = h;
        width = w;
    }

    public void SetNumChannels(int i)
    {
        numChannels = Mathf.Max(1, i);
    }
    
    private void GetProperties()
    {
        useSensor = boxcastSensorProperties.useSensor;
        SetDepthHeightWidth(boxcastSensorProperties.depth,boxcastSensorProperties.height,boxcastSensorProperties.width);
        SetNumChannels(boxcastSensorProperties.numChannels);
    }

    /// <summary>
    /// Sets the properties of the CameraSensor from values
    /// </summary>
    public override void SetProperties()
    {
        InitializeSensor();
        
        if (!boxcastSensorComponent)
        {
            return;
        }
        
        boxcastSensorComponent.observationStacks = observationStacks;
        boxcastSensorComponent.numChannels = numChannels;
        boxcastSensorComponent.SetSampleSize(depth,height,width);
        boxcastSensorComponent.CreateSensors();
    }
}
