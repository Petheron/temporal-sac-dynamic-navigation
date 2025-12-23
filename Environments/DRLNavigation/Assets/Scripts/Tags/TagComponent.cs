using UnityEngine;

public class TagComponent : MonoBehaviour
{
    public VisionTag detectableTag;
    public bool assignToAllChildren = false;

    [Header("Testing")] 
    private int testValue;
    private int[] testTagArray;

    public void AssignTagToGameobject(Transform root)
    {
        VisibleTags.AssignTag(root.gameObject,detectableTag);
        if (assignToAllChildren)
        {
            for (int i = 0; i < root.childCount; i++)
            {
                var childTransform = transform.GetChild(i);
                VisibleTags.AssignTag(childTransform.gameObject, detectableTag);
            }
        }
    }
}
