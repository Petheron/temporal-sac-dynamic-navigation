using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;


[CustomEditor((typeof(Digraph)))]
[CanEditMultipleObjects]
public class DigraphEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        var so = serializedObject;
        so.Update();

        if (GUILayout.Button("Update Graph"))
        {
            UpdateGraph();
        }
        
        if (GUILayout.Button("Get Shortest Path"))
        {
            ShortestPath();
        }

        if (GUILayout.Button("Save Vertices to JSON"))
        {
            SaveVertices();
        }
        if (GUILayout.Button("Load Vertices from JSON"))
        {
            LoadVertices();
        }
    }

    private void SaveVertices()
    {
        var digraph = serializedObject.targetObject as Digraph;
        if (digraph)
        {
            digraph.SaveVertices();
        }
    }
    
    private void LoadVertices()
    {
        var digraph = serializedObject.targetObject as Digraph;
        if (digraph)
        {
            digraph.LoadVertices();
        }
    }

    private void UpdateGraph()
    {
        var digraph = serializedObject.targetObject as Digraph;
        if (digraph)
        {
            digraph.SetMinMaxDistances(digraph.minimumDistance, digraph.maximumDistance);
            digraph.UpdateEdges();
        }
    }
    
    private void ShortestPath()
    {
        var digraph = serializedObject.targetObject as Digraph;
        if (digraph)
        {
            digraph.shortestPath = new List<Vertex>();
            digraph.shortestPath = digraph.GetShortestPath(digraph.start, digraph.goal);
        }
    }

}
