using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;


[CustomEditor((typeof(VertexSamplingManager)))]
[CanEditMultipleObjects]
public class NavMeshManagerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        var so = serializedObject;
        so.Update();

        if (GUILayout.Button("Reset Graph"))
        {
            ResetGraph();
        }
        
        if (GUILayout.Button("Update Edges"))
        {
            UpdateEdges();
        }
        
        if (GUILayout.Button("Prune Edges"))
        {
            PruneEdges();
        }
    }

    private void ResetGraph()
    {
        var navMeshManager = serializedObject.targetObject as VertexSamplingManager;
        if (navMeshManager)
        {
            navMeshManager.CreateGraph();
        }
    }
    
    private void UpdateEdges()
    {
        var navMeshManager = serializedObject.targetObject as VertexSamplingManager;
        if (navMeshManager)
        {
            navMeshManager.UpdateEdges();
        }
    }
    
    private void PruneEdges()
    {
        var navMeshManager = serializedObject.targetObject as VertexSamplingManager;
        if (navMeshManager)
        {
            navMeshManager.PruneEdges();
        }
    }

}
