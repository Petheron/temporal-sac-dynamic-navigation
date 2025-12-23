using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Editor;

[CustomEditor(typeof(BoxcastSensorComponent), editorForChildClasses: true)]
[CanEditMultipleObjects]
public class BoxcastSensorComponentEditor : Editor
{
    public override void OnInspectorGUI()
    {
        var so = serializedObject;
        so.Update();
        
        EditorGUI.BeginChangeCheck();
        
        EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
        {
            EditorGUILayout.PropertyField(so.FindProperty(nameof(BoxcastSensorComponent.sensorName)), true);

            EditorGUILayout.LabelField("Grid Settings", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(so.FindProperty(nameof(BoxcastSensorComponent.sampleSize)), true);
            EditorGUILayout.LabelField("Sensor Settings", EditorStyles.boldLabel);
            EditorGUILayout.ObjectField(so.FindProperty(nameof(BoxcastSensorComponent.agentLocationGameObject)));
            EditorGUILayout.PropertyField(so.FindProperty(nameof(BoxcastSensorComponent.observationStacks)), true);
            EditorGUILayout.LabelField("Testing Settings", EditorStyles.boldLabel);
        }
        EditorGUI.EndDisabledGroup();
        
        EditorGUILayout.PropertyField(so.FindProperty(nameof(BoxcastSensorComponent.testChannelIndex)), true);
        EditorGUILayout.PropertyField(so.FindProperty(nameof(BoxcastSensorComponent.drawGizmos)), true);


        var requireSensorUpdate = EditorGUI.EndChangeCheck();
        so.ApplyModifiedProperties();

        if (requireSensorUpdate)
        {
            UpdateSensor();
        }
    }
    
    void UpdateSensor()
    {
        var sensorComponent = serializedObject.targetObject as BoxcastSensorComponent;
        sensorComponent?.UpdateSensor();
    }

}
