using UnityEngine;
using UnityEditor;

[CustomEditor((typeof(TrajectoryComponent)))]
[CanEditMultipleObjects]
public class TrajectoryComponentEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        var so = serializedObject;
        so.Update();

        if (GUILayout.Button("Get Trajectory Information"))
        {
            GetTrajectoryInformation();
        }
    }

    private void GetTrajectoryInformation()
    {
        var trajectoryComponent = serializedObject.targetObject as TrajectoryComponent;
        if (trajectoryComponent)
        {
            Debug.Log(trajectoryComponent.ToString());
        }
    }
}
