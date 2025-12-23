using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.MLAgents;

[AddComponentMenu("ML Agents/Boxcast Sensor")]
/// <summary>
/// A SensorComponent that creates a <see cref="BoxcastSensorBase"/>.
/// Creates the sensors needed for Boxcast detection.
/// </summary>
public class BoxcastSensorComponent : SensorComponent
{
    private List<BoxcastSensor> sensors;

    private BoxcastSensor debugSensor;

    private OccupancyMapPerception boxcastPerception;
    
    [Header("Sensor Variables")]
    
    public string sensorName = "BoxcastSensor";

    public Vector3Int sampleSize = new Vector3Int(3,3,3);

    public int observationStacks;

    public int numChannels = 1;

    public GameObject agentLocationGameObject;

    public OccupancyMap occupancyMap;

    [Header("Testing Variables")] 
    public int testChannelIndex = 0;
    public bool drawGizmos = true;

    public override ISensor[] CreateSensors()
    {
        if (agentLocationGameObject == null)
        {
            agentLocationGameObject = gameObject;
        }

        if (occupancyMap == null)
        {
            occupancyMap = FindObjectOfType<OccupancyMap>();
        }
        
        boxcastPerception = new OccupancyMapPerception(sampleSize, numChannels, agentLocationGameObject.transform, occupancyMap);

        sensors = GetBoxcastSensors().ToList();

        debugSensor = new BoxcastSensor("Debug Boxcast Sensor", sampleSize, numChannels);
        boxcastPerception.RegisterDebugSensor(debugSensor);

        if (sensors == null || sensors.Count < 1)
        {
            throw new UnityAgentsException("BoxcastComponent received no sensors.");
        }

        sensors[0].gridPerception = boxcastPerception;
        foreach (var sensor in sensors)
        {
            boxcastPerception.RegisterSensor(sensor);
        }

        if (observationStacks > 1)
        {
            var sensors = new ISensor[this.sensors.Count];
            for (var i = 0; i < this.sensors.Count; i++)
            {
                sensors[i] = new StackingSensor(this.sensors[i], observationStacks);
            }
            return sensors;
        }
        else
        {
            return sensors.ToArray();
        }
    }

    private BoxcastSensor[] GetBoxcastSensors()
    {
        List<BoxcastSensor> sensorList = new List<BoxcastSensor>();
        var sensor = new BoxcastSensor(sensorName, sampleSize*Vector3Int.one, numChannels);
        sensorList.Add(sensor);
        return sensorList.ToArray();
    }

    /// <summary>
    /// Update the properties of sensor or perception that have been changed in the editor if necessary. 
    /// </summary>
    public void UpdateSensor()
    {
    }

    public void SetSampleSize(int depth, int height, int width)
    {
        sampleSize = BaseGrid.GetOddGrid(new Vector3Int(width, depth, height));
    }
    
#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        if (!drawGizmos) return;
        
        if (occupancyMap == null || debugSensor == null)
        {
            return;
        }

        if (debugSensor.PerceptionBuffer == null)
        {
            return;
        }

        var buffer = debugSensor.PerceptionBuffer;
        var indicesBuffer = debugSensor.IndicesBuffer;
        var cellSize = occupancyMap.CellSize(testChannelIndex);
        var orientation = occupancyMap.GetGridOrientation();

        if (buffer == null) return;
        if (indicesBuffer == null) return;

        Debug.Assert(testChannelIndex>=0 && testChannelIndex<numChannels);
        
        for (int i = testChannelIndex; i < indicesBuffer.Length; i+=numChannels)
        {
            int idx = indicesBuffer[i];
            int value = buffer[i];
            Vector3 position = occupancyMap.GetCellGlobalPosition(idx, testChannelIndex);
            
            Color cellCol = occupancyMap.occupancyGridProperties.GetColor(value);

            Gizmos.matrix = Matrix4x4.TRS(Vector3.zero,orientation,Vector3.one);
            Gizmos.color = cellCol;
            Gizmos.DrawCube(position, cellSize*Vector3.one);
            Gizmos.matrix = Matrix4x4.identity;
        }
    }
#endif
}
