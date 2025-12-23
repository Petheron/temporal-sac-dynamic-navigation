using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TrajectoryComponent : MonoBehaviour
{
    private Trajectory trajectory;
    private LineRenderer lineRenderer;
    public LineRendererProperties lineRendererProperties;
    public GameObject startCube;
    public GameObject goalCube;
    public List<GameObject> complexActionPrefabList;
    [SerializeField] private AgentType agentType;

    [SerializeField] private int startIndex;
    [SerializeField] private int goalIndex;
    private static readonly int DashCount = Shader.PropertyToID("_DashCount");


    private void Awake()
    {
        lineRenderer = GetComponent<LineRenderer>();
    }

    public void VisualizeTrajectory(Trajectory newTrajectory)
    {
        if (newTrajectory == null) return;
        if (!lineRenderer) return;
        if (!lineRendererProperties) return;

        trajectory = newTrajectory;
        startIndex = trajectory.startIndex;
        goalIndex = trajectory.goalIndex;
        agentType = trajectory.agentType;

        name = $"{trajectory.trajectoryType.ToString()}_trajectory_{trajectory.ID}";

        var positions = trajectory.positions.ToArray();
        lineRenderer.positionCount = positions.Length;
        lineRenderer.SetPositions(positions);
        
        Color trajectoryColor = lineRendererProperties.GetTrajectoryColor(trajectory);
        lineRenderer.material.color = trajectoryColor;
        lineRenderer.material.SetInt(DashCount,(int)agentType);
        lineRendererProperties.SetLineRendererParameters(ref lineRenderer);

        if (startCube)
        {
            startCube.transform.position = positions[0];
            startCube.name = "Start";
            startCube.GetComponent<Renderer>().material.color = trajectoryColor;
        }
        
        if (goalCube)
        {
            goalCube.transform.position = trajectory.goalPosition + Vector3.up;
            goalCube.name = "Goal";
            goalCube.GetComponent<Renderer>().material.color = trajectoryColor;
        }
        
        if (trajectory.complexActionPositions != null)
        {
            foreach (var actionPosPair in trajectory.complexActionPositions)
            {
                string actionName = actionPosPair.Key.ToString();
                GameObject actionContainerGO = new GameObject
                {
                    transform = { parent = transform },
                    name = $"{actionName}Positions"
                };

                GameObject prefab = complexActionPrefabList[(int) actionPosPair.Key];
        
                for (int i = 0; i < actionPosPair.Value.Count; i++)
                {
                    var pos = actionPosPair.Value[i];
                    GameObject actionGO = Instantiate(prefab, pos, Quaternion.identity, actionContainerGO.transform);
                    actionGO.name = $"{actionName}_{i}";
                    // actionGO.GetComponent<Renderer>().material.color = lineRendererProperties.jumpColor;
                }
            }
        }
    }

    public bool IsVisibleByIndex(int visibleStartIndex, int visibleGoalIndex)
    {
        if (visibleStartIndex < 0 && visibleGoalIndex < 0)
        {
            return true;
        }
        return (startIndex == visibleStartIndex || visibleStartIndex < 0) && 
               (goalIndex == visibleGoalIndex || visibleGoalIndex < 0);
    }

    public bool IsVisibleByAgentType(VisibleAgentType visibleAgentType)
    {
        if (visibleAgentType == VisibleAgentType.All)
        {
            return true;
        }
        return agentType.ToString() == visibleAgentType.ToString();
        
    }

    public override string ToString()
    {
        if (trajectory == null)
        {
            return "Trajectory is null";
        }
        return trajectory.ToString();
    }
}
