using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using Unity.Collections.LowLevel.Unsafe;
using Unity.MLAgents.Sensors;
using Unity.MLAgents;
using UnityEngine.UI;

[AddComponentMenu("ML Agents/Depthmap Sensor")]
public class DepthmapSensorComponent : SensorComponent
{
    private DepthmapSensor m_Sensor;
    private DepthmapSensor m_DebugSensor;
    private DepthmapPerception m_DepthmapPerception;

    public string sensorName = "DepthmapSensor";
    public int height = 8;
    public int width = 8;
    public int observationStacks = 1;
    public LayerMask layerMask;
    public VisibleTags visibleTags;
    public int numChannels = 1;
    public float maxAngle = 60f;
    public float maxDistance = 10f;
    [Header("Debugging")]
    public bool showGizmos;
    public int debugChannelIdx;
    public RawImage debugImage;

    public override ISensor[] CreateSensors()
    {
        Dispose();

        int tagsCount = visibleTags?.GetVisibleTagsCount() ?? 0;
        int modifiedNumChannels = numChannels<=0 || numChannels>tagsCount ? tagsCount: numChannels;

        m_DepthmapPerception = new DepthmapPerception(height,width,maxAngle,maxDistance,layerMask, visibleTags, modifiedNumChannels, transform);

        m_Sensor = new DepthmapSensor(sensorName,height,width, modifiedNumChannels);
        m_DebugSensor = new DepthmapSensor("DepthmapDebugSensor",height,width, modifiedNumChannels);

        m_Sensor.depthmapPerception = m_DepthmapPerception;
        
        m_DepthmapPerception.RegisterSensor(m_Sensor);
        m_DepthmapPerception.RegisterDebugSensor(m_DebugSensor);
        
        if (observationStacks > 1)
        {
            return new ISensor[] { new StackingSensor(m_Sensor, observationStacks) };
        }

        return new ISensor[] {m_Sensor};
    }

    private void FixedUpdate()
    {
        if (!debugImage) return;
        if (!debugImage.canvas.worldCamera.enabled) return;
        debugImage.texture = m_DebugSensor?.PerceptionTexture(debugChannelIdx);
    }

    public void UpdateSensor()
    {
        // Leave empty
    }

    public void Dispose()
    {
        if (!ReferenceEquals(null, m_Sensor))
        {
            m_Sensor.Dispose();
            m_Sensor = null;
        }
    }
    
#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        if (!showGizmos) return;
        
        // Draw texture for debugging
        if (m_DebugSensor != null)
        {
            Gizmos.matrix = Matrix4x4.Translate(40f*Vector3.up);
            var texture = m_DebugSensor.PerceptionTexture(debugChannelIdx);
            Gizmos.DrawGUITexture(new Rect(0f,40f,10f,10f),texture);
            Gizmos.matrix = Matrix4x4.identity;
        }

        // Draw rays to show vision bounds
        var t = transform;
        Gizmos.matrix = Matrix4x4.TRS(t.position,Quaternion.identity,Vector3.one);
        if (m_DepthmapPerception!=null)
        {
            var rayLengths = m_DepthmapPerception.debugRayLengths;
            var rayDirections = m_DepthmapPerception.debugRayDirections;
            for (int i = 0; i < rayLengths.Length; i++)
            {
                Gizmos.DrawLine(Vector3.zero,rayLengths[i]*rayDirections[i]);
            }
        }
        Gizmos.matrix = Matrix4x4.identity;
    }
#endif


}
