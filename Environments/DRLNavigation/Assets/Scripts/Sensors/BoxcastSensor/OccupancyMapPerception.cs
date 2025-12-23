using System.Collections;
using System.Collections.Generic;
using System;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class OccupancyMapPerception : IBoxcastPerception
{
    private OccupancyMap occupancyMap;
    private Vector3Int sampleSize;
    private int numChannels = 1;
    private Transform agentTransform;

    public event Action<int[]> BoxcastSample;
    public event Action<int[]> DebugBoxcastSample;
    public event Action<int[]> DebugBoxcastIndices;
    
    public OccupancyMapPerception
    (
        Vector3Int sampleSize,
        int numChannels,
        Transform agentTransform,
        OccupancyMap occupancyMap
    )
    {
        this.sampleSize = sampleSize;
        this.numChannels = numChannels;
        this.agentTransform = agentTransform;
        this.occupancyMap = occupancyMap;
    }

    public Vector3 GetCellGlobalPosition(int cellIndex)
    {
        if (!occupancyMap) return Vector3.zero;
        return occupancyMap.GetCellGlobalPosition(cellIndex);
    }

    public void Perceive()
    {
        if (!agentTransform) return;
        if (!occupancyMap) return;

        Vector3 position = agentTransform.position;

        // sample size should already be corrected to be odd.
        int numSamples = sampleSize.x * sampleSize.y * sampleSize.z;
        Debug.Assert(numSamples%2==1);

        occupancyMap.SampleOccupancyMap(position, sampleSize, out int[] samples, out int[] indices, 0, numChannels);

        BoxcastSample?.Invoke(samples);
        DebugBoxcastSample?.Invoke(samples);
        DebugBoxcastIndices?.Invoke(indices);
    }

    public void UpdateGizmo()
    {
        
    }

    public void RegisterSensor(BoxcastSensor sensor)
    {
        BoxcastSample += sensor.ProcessSampleArray;
    }


    public void RegisterDebugSensor(BoxcastSensor debugSensor)
    {
        DebugBoxcastSample += debugSensor.ProcessSampleArray;
        DebugBoxcastIndices += debugSensor.ProcessIndicesArray;
    }
}
