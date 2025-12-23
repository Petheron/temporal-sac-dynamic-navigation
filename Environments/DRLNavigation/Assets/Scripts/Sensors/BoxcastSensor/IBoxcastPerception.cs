using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

public interface IBoxcastPerception
{
    Vector3 GetCellGlobalPosition(int cellIndex);

    void Perceive();
    
    void UpdateGizmo();

    void RegisterSensor(BoxcastSensor sensor);

    void RegisterDebugSensor(BoxcastSensor debugSensor);
}
