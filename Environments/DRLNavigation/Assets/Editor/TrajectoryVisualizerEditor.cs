using UnityEngine;
using UnityEditor;

[CustomEditor((typeof(TrajectoryVisualizer)))]
[CanEditMultipleObjects]
public class TrajectoryVisualizerEditor : Editor
{
    private bool requiresUpdate;
    
    public override void OnInspectorGUI()
    {
        EditorGUI.BeginChangeCheck();
        base.OnInspectorGUI();

        var so = serializedObject;
        so.Update();

        if (EditorGUI.EndChangeCheck())
        {
            requiresUpdate = true;
        }
        
        PerformDirtyUpdate();
        
        
        
        // if (GUILayout.Button("Update Visualized Trajectories"))
        // {
        //     UpdateTrajectoryVisualization();
        // }
    }

    private void PerformDirtyUpdate()
    {
        if (!requiresUpdate) return;
        UpdateTrajectoryVisualization();
        requiresUpdate = false;
    }

    private void UpdateTrajectoryVisualization()
    {
        var trajectoryVisualizer = serializedObject.targetObject as TrajectoryVisualizer;
        if (trajectoryVisualizer)
        {
            trajectoryVisualizer.ToggleTrajectoryVisibility();
        }
    }
}
