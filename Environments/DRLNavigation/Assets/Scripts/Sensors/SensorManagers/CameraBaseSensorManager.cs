using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class CameraBaseSensorManager : BaseSensorManager
{
    private CameraSensorComponent m_CameraSensorComponent;

    [SerializeField] private int m_Width = 84;
    [SerializeField] private int m_Height=84;
    [SerializeField] private bool m_GrayScale=false;

    
    public override void Initialize(SensorManagerProperties properties)
    {
        observationStacks = properties.ObservationStacks;
        SetProperties();
    }
    
    protected override void InitializeSensor()
    {
        m_CameraSensorComponent = GetComponent<CameraSensorComponent>();
        if (!m_CameraSensorComponent)
        {
            m_CameraSensorComponent = GetComponentInChildren<CameraSensorComponent>();
        }
        environmentParameters = Academy.Instance.EnvironmentParameters;

        if (environmentParameters != null)
        {
            StoreProperties();
        }

        if (m_CameraSensorComponent)
        {
            if ((m_Width <= 0 || m_Height <= 0) || !useSensor)
            {
                m_CameraSensorComponent.Dispose();
                DestroyImmediate(m_CameraSensorComponent);
            }
        }
        
    }

    protected override void StoreProperties()
    {
        m_Width = (int) environmentParameters.GetWithDefault("width", m_Width);
        m_Height = (int) environmentParameters.GetWithDefault("height ", m_Height);
        m_GrayScale = environmentParameters.GetWithDefault("grayscale", m_GrayScale?1:0) > 0.0f;
        observationStacks= (int) environmentParameters.GetWithDefault("observation_stacks", observationStacks);
    }

    /// <summary>
    /// Sets the properties of the CameraSensor from values
    /// </summary>
    public override void SetProperties()
    {
        InitializeSensor();
        
        if (!m_CameraSensorComponent)
        {
            return;
        }

        m_CameraSensorComponent.Height = m_Height;
        m_CameraSensorComponent.Width = m_Width;
        m_CameraSensorComponent.Grayscale = m_GrayScale;
        m_CameraSensorComponent.ObservationStacks = observationStacks;
        m_CameraSensorComponent.CreateSensors();
    }
}
