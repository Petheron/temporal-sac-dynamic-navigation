using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "LineRendererProperties", menuName = "Line Renderer Properties", order = 0)]
public class LineRendererProperties : ScriptableObject
{
    [Header("LineRenderer Properties")]
    public float lineWidth = 0.2f;
    public int numCornerVertices = 4;
    public int numCapVertices = 3;
    public float simplifyTolerance = 0.75f;
    
    [Header("Trajectory Color Settings")]
    public Color successColor = Color.blue;
    public Color failureColor = Color.yellow;
    public Color timeOutColor = Color.red;
    public Color eucFailColor = Color.green;
    public Color navmeshColor = new(1.0f,0f,1.0f);
    public Color jumpColor = new(1.0f, 0.25f, 0f);
    [Tooltip("Controls dash density. Higher number means smaller dashes along trajectory.")]public float dashDensity = 25.0f;
    private static readonly int DashDensity = Shader.PropertyToID("_DashDensity");

    public void SetLineRendererParameters(ref LineRenderer lineRenderer)
    {
        lineRenderer.startWidth = lineWidth;
        lineRenderer.Simplify(simplifyTolerance);
        lineRenderer.numCornerVertices = numCornerVertices;
        lineRenderer.numCapVertices = numCapVertices;
        lineRenderer.material.SetFloat(DashDensity, dashDensity);
    }
    
    public Color GetTrajectoryColor(Trajectory trajectory)
    {
        if (trajectory.trajectoryType == TrajectoryType.Navmesh)
        {
            return navmeshColor;
        }
        switch (trajectory.endCondition)
        {
            case EndCondition.Success:
                return successColor;
            case EndCondition.Failure:
                return failureColor;
            case EndCondition.Timeout:
                return timeOutColor;
            case EndCondition.EucFail:
                return eucFailColor;
            default:
                return Color.white;
        }
    }
    
}
