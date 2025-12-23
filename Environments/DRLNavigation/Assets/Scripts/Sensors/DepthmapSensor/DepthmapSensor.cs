using System;
using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.Profiling;

public class DepthmapSensor : ISensor, IDisposable
{
    private string name;
    private ObservationSpec observationSpec;
    private int height;
    private int width;
    private int ArrayLength => height*width*numChannels;
    private int numChannels;

    public DepthmapPerception depthmapPerception;

    private float[] m_PerceptionBuffer;
    private Texture2D m_PerceptionTexture;

    public DepthmapSensor(string sensorName, int sensorHeight, int sensorWidth, int numSensorChannels)
    {
        name = sensorName;
        height = sensorHeight;
        width = sensorWidth;
        numChannels = numSensorChannels;
        observationSpec = ObservationSpec.Visual(height, width, numChannels);

        ResetPerceptionBuffer();
        m_PerceptionTexture = new Texture2D(width, height, TextureFormat.RGB24,false);
        m_PerceptionTexture.filterMode = FilterMode.Point;
    }

    public float[] PerceptionBuffer()
    {
        return m_PerceptionBuffer;
    }

    public Texture2D PerceptionTexture(int channelIndex)
    {
        if (m_PerceptionBuffer == null) return m_PerceptionTexture;
        int c = Mathf.Min(Mathf.Max(0, channelIndex),numChannels-1);
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                var v = m_PerceptionBuffer[GetArrayPosition(h,w,c)];
                Color color = new Color(v, v, v);
                m_PerceptionTexture.SetPixel(w, h, color);
            }
        }
        m_PerceptionTexture.Apply();
        return m_PerceptionTexture;
    }

    private void ResetPerceptionBuffer()
    {
        if (m_PerceptionBuffer != null)
        {
            Array.Clear(m_PerceptionBuffer,0,m_PerceptionBuffer.Length);
        }
        else
        {
            m_PerceptionBuffer = new float[ArrayLength];
        }
    }

    public ObservationSpec GetObservationSpec()
    {
        return observationSpec;
    }

    public void ProcessSampleArray(float[] sampleArray)
    {
        Array.Copy(sampleArray,0,m_PerceptionBuffer,0,sampleArray.Length);
    }

    public int Write(ObservationWriter writer)
    {
        int index = 0;
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                for (int c = 0; c < numChannels; c++)
                {
                    writer[h, w, c] = m_PerceptionBuffer[index];
                    index++;
                }
            }
        }
        return index;
    }

    private int GetArrayPosition(int h, int w, int c)
    {
        if (h < 0 || h >= height) return -1;
        if (w < 0 || w >= width) return -1;
        if (c < 0 || c >= numChannels) return -1;
        return h * width * numChannels + w * numChannels + c;
    }

    public byte[] GetCompressedObservation()
    {
        return new List<byte>().ToArray();
    }

    public void Update()
    {
        ResetPerceptionBuffer();

        depthmapPerception?.Perceive();
    }

    public void Reset()
    {
        // Leave empty
    }

    public CompressionSpec GetCompressionSpec()
    {
        return new CompressionSpec(SensorCompressionType.None);
    }

    public string GetName()
    {
        return name;
    }

    public void Dispose()
    {
        if (!ReferenceEquals(null, m_PerceptionTexture))
        {
            if (Application.isEditor)
            {
                // Edit Mode tests complain if we use Destroy()
                UnityEngine.Object.DestroyImmediate(m_PerceptionTexture);
            }
            else
            {
                UnityEngine.Object.Destroy(m_PerceptionTexture);
            }
            m_PerceptionTexture = null;
        }
    }
}
