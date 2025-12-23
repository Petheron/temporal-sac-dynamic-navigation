using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;


[CustomEditor((typeof(TagComponent)))]
[CanEditMultipleObjects]
public class AssignTagEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        var so = serializedObject;
        so.Update();

        if (GUILayout.Button("Assign Tag"))
        {
            AssignTag();
        }
    }

    private void AssignTag()
    {
        var assignTag = serializedObject.targetObject as TagComponent;
        if (assignTag)
        {
            assignTag.AssignTagToGameobject(assignTag.transform);
        }
    }

}
