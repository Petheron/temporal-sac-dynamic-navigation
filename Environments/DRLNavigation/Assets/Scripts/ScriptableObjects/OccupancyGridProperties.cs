using System;
using Unity.VisualScripting.FullSerializer;
using UnityEngine;

public enum OccupancyGridDataType{
    VolumetricGrid,
    Octree,
}

[CreateAssetMenu(fileName = "OccupancyGridProperties", menuName = "Occupancy Grid Properties", order = 0)]
public class OccupancyGridProperties : ScriptableObject
{
    [Header("Grid Properties")] 
    public OccupancyGridDataType occupancyGridDataType = OccupancyGridDataType.VolumetricGrid;
    public int gridSize = 128;
    public float cellSize = 1.0f;
    public int levelsOfDetail = 1;
    public float cellSizePercentageBuffer = 1f;
    public bool useTagIndicesForVoxelValue = true;
    public LayerMask colliderMask;
    public VisionTag visionTags;
    public MutatorType visibleMutatorTypes;
    public VisibleTags VisibleTags
    {
        get
        {
            if (visibleTags == null)
            {
                visibleTags = new VisibleTags(visionTags, visibleMutatorTypes);
            }
            else
            {
                if (visionTags != visibleTags.GetVisionTags() || visibleMutatorTypes != visibleTags.GetVisibleMutatorTypes())
                {
                    visibleTags.UpdateTagData(visionTags, visibleMutatorTypes);
                }
            }
            return visibleTags;
        }
    }
    private VisibleTags visibleTags;
    

    [Header("Rotator Properties")] 
    public bool rotateSampleWithAgent;
    public int maxRotatorSampleSize = 30;
    public bool correctForbiddenAngles;
    public bool forceToCubeEdge;
    public bool scaleRadius;
    

    [Header("Gizmo Properties")]
    public Color untaggedColor = new Color(0f, 0f, 1f);
    public Color solidColor = new Color(1f, 0f, 0f);
    public Color dynamicsColor = new Color(1f, 1f, 0f);
    public Color affectorColor = new Color(1f, 0f, 1f);
    public Color affectedColor = new Color(1f, 0.5f, 0.75f);
    public Color agentColor = new Color(0f, 1f, 0f);
    public Color dangerColor = new Color(1f, 0.5f, 0f);
    public Color mutatorColor = new Color(1f, 1f, 1f);
    
    [Range(0,1)]public float volumeAlpha = 0.05f;
    [SerializeField, Range(0,1)] private float untaggedAlpha;
    [Range(0,1)]public float edgeAlpha = 0.5f;


    private int Pow(int b, int power)
    {
        if (power < 0) return 0;
        int v = 1;
        for (int i = 0; i < power; i++)
        {
            v *= b;
        }
        return v;
    }
    public Color GetColor(int value)
    {
        Color color;
        int passedValue = value;
        if (useTagIndicesForVoxelValue)
        {
            passedValue = Pow(2, value-1);
        }

        switch (passedValue)
        {
            case (int)VisionTag.Solid:
                color =  solidColor;
                break;
            case (int)VisionTag.Independent:
                color = dynamicsColor;
                break;
            case (int)VisionTag.Affector:
                color = affectorColor;
                break;
            case (int)VisionTag.Affected:
                color = affectedColor;
                break;
            case (int)VisionTag.Agent:
                color = agentColor;
                break;
            case (int)VisionTag.Danger:
                color = dangerColor;
                break;
            case 0:
                color = untaggedColor;
                break;
            default:
                color = mutatorColor;
                break;
        }
        
        color.a = color == untaggedColor ? untaggedAlpha : volumeAlpha;
        return color;
    }
}
