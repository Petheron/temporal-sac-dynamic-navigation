using UnityEngine;
using Unity.MLAgents;

public class DepthmapSensorManager : BaseSensorManager
{
    public DepthMapSensorProperties depthMapSensorProperties;
    private DepthmapSensorComponent m_DepthmapSensorComponent;
    
    [ReadOnly] public int width = 8;
    [ReadOnly] public int height=8;
    [ReadOnly] public float maxDistance=10;
    [ReadOnly] public float maxAngle=60;
    [ReadOnly] public LayerMask layerMask;
    [ReadOnly] public VisibleTags visibleTags;
    [ReadOnly] public VisionTag visionTags;
    [ReadOnly] public MutatorType visibleMutatorTypes;
    [ReadOnly] public int numChannels = 1;

    
    public override void Initialize(SensorManagerProperties properties)
    {
        observationStacks = properties.ObservationStacks;
        SetProperties();
    }
    
    protected override void InitializeSensor()
    {
        m_DepthmapSensorComponent = GetComponent<DepthmapSensorComponent>();
        if (!m_DepthmapSensorComponent)
        {
            m_DepthmapSensorComponent = GetComponentInChildren<DepthmapSensorComponent>();
        }

        if (!m_DepthmapSensorComponent)
        {
            DestroyImmediate(this);
            return;
        }

        if (depthMapSensorProperties)
        {
            GetProperties();
        }
        
        environmentParameters = Academy.Instance.EnvironmentParameters;
        if (environmentParameters != null)
        {
            StoreProperties();
        }

        if (m_DepthmapSensorComponent)
        {
            if ((!useSensor || width <= 0 || height <= 0))
            {
                m_DepthmapSensorComponent.Dispose();
                DestroyImmediate(m_DepthmapSensorComponent);
                return;
            }
        }
    }

    private void GetProperties()
    {
        useSensor = depthMapSensorProperties.useSensor;
        width = depthMapSensorProperties.width;
        height = depthMapSensorProperties.height;
        maxDistance = depthMapSensorProperties.maxRayLength;
        maxAngle = depthMapSensorProperties.maxAngle;
        layerMask = depthMapSensorProperties.layerMask;
        visibleTags = depthMapSensorProperties.VisibleTags;
        numChannels = Mathf.Max(depthMapSensorProperties.numChannels,1);
    }

    protected override void StoreProperties()
    {
        useSensor = environmentParameters.GetWithDefault("use_depthmap_sensor", useSensor?1:0) > 0.0f;
        width = (int) environmentParameters.GetWithDefault("depthmap_width", width);
        height = (int) environmentParameters.GetWithDefault("depthmap_height", height);
        maxDistance = environmentParameters.GetWithDefault("depthmap_ray_length", maxDistance);
        maxAngle = environmentParameters.GetWithDefault("depthmap_max_angle", maxAngle);
        
        int visionTagMask = (int) environmentParameters.GetWithDefault("visible_layer_mask",-1);
        int mutatorTypeMask = (int) environmentParameters.GetWithDefault("visible_mutator_mask",-1);
        
        visibleTags.SetDataFromBitMasks(visionTagMask, mutatorTypeMask);
        visionTags = visibleTags.GetVisionTags();
        visibleMutatorTypes = visibleTags.GetVisibleMutatorTypes();
        
        numChannels = (int)environmentParameters.GetWithDefault("depthmap_num_channels", numChannels);

        observationStacks= (int) environmentParameters.GetWithDefault("observation_stacks", observationStacks);
    }

    /// <summary>
    /// Sets the properties of the CameraSensor from values
    /// </summary>
    public override void SetProperties()
    {
        InitializeSensor();
        
        if (!m_DepthmapSensorComponent)
        {
            return;
        }

        m_DepthmapSensorComponent.height = height;
        m_DepthmapSensorComponent.width = width;
        m_DepthmapSensorComponent.maxDistance = maxDistance;
        m_DepthmapSensorComponent.maxAngle = maxAngle;
        m_DepthmapSensorComponent.observationStacks = observationStacks;
        m_DepthmapSensorComponent.layerMask = layerMask;
        m_DepthmapSensorComponent.visibleTags = visibleTags;
        m_DepthmapSensorComponent.numChannels = numChannels;
        m_DepthmapSensorComponent.CreateSensors();
    }
}
