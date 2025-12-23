using System;
using UnityEngine;

[CreateAssetMenu(fileName = "BoxcastSensorProperties", menuName = "Boxcast Sensor Properties", order = 0)]
public class BoxcastSensorProperties : ScriptableObject
{
    public bool useSensor = true;
    public int depth = 9;
    public int width = 9;
    public int height = 9;
    public int numChannels = 1;
}
