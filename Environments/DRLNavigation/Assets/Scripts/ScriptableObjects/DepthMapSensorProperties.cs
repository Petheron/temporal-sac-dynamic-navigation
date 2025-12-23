using System;
using UnityEngine;

[CreateAssetMenu(fileName = "DepthMapSensorProperties", menuName = "Depth Map Sensor Properties", order = 0)]
public class DepthMapSensorProperties : ScriptableObject
{
    [Header("Grid Properties")] 
    public bool useSensor = true;
    public int width = 9;
    public int height = 9;
    public float maxRayLength = 20.0f;
    public float maxAngle = 45.0f;
    public LayerMask layerMask;
    public int numChannels=0;
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
}
