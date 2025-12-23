using UnityEngine;
using UnityEditor;


[CustomEditor((typeof(OccupancyMap)))]
[CanEditMultipleObjects]
public class OccupancyMapEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        var so = serializedObject;
        so.Update();

        if (GUILayout.Button("Initialize Occupancy Map"))
        {
            InitializeOccupancyMapEditor();
        }
        if (GUILayout.Button("Sample"))
        {
            Sample();
        }
        if (GUILayout.Button("Test Location Index"))
        {
            TestLocationIndex();
        }
        // if (GUILayout.Button("Get ordered hinge angles"))
        // {
        //     GetSortedHingeAngles();
        // }
        // if (GUILayout.Button("Get non-forbidden angle"))
        // {
        //     GetNonForbiddenAngle();
        // }
        // if (GUILayout.Button("Get Lower bound hinge angle"))
        // {
        //     GetLowerHingeAngle();
        // }
        
    }

    // private void GetLowerHingeAngle()
    // {
    //     var occupancyMap = serializedObject.targetObject as OccupancyMap;
    //     if (occupancyMap)
    //     {
    //         float angle = occupancyMap.testSampleEulerRotation.y * -1f;
    //         occupancyMap.GetLowerHingeAngleEditor(angle);
    //     }
    // }
    //
    // private void GetNonForbiddenAngle()
    // {
    //     var occupancyMap = serializedObject.targetObject as OccupancyMap;
    //     if (occupancyMap)
    //     {
    //         float angle = occupancyMap.testSampleEulerRotation.y * -1f;
    //         Debug.Log($"Angle :{angle} --> {occupancyMap.GetNonForbiddenAngle(angle,1)}"); ;
    //         
    //     }
    // }
    //
    // private void GetSortedHingeAngles()
    // {
    //     var occupancyMap = serializedObject.targetObject as OccupancyMap;
    //     if (occupancyMap)
    //     {
    //         var list = occupancyMap.GetOrderedHingeAngles();
    //         foreach (var l in list)
    //         {
    //             Debug.Log($"{l} with angle {l.Angle()} with term {l.GetComparisonTerm()}");
    //         }
    //     }
    // }

    private void InitializeOccupancyMapEditor()
    {
        var occupancyMap = serializedObject.targetObject as OccupancyMap;
        if (occupancyMap)
        {
            occupancyMap.InitializeOccupancyMap();
        }
        
    }

    private void Sample()
    {
        var occupancyMap = serializedObject.targetObject as OccupancyMap;
        if (occupancyMap)
        {
            occupancyMap.TestSampling();
        }
    }

    private void TestLocationIndex()
    {
        var occupancyMap = serializedObject.targetObject as OccupancyMap;
        if (occupancyMap)
        {
            occupancyMap.TestIndexFromLocation();
        } 
    }
}
