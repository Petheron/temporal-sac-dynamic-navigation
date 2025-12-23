using LevelGeneration;
using UnityEngine;
using UnityEditor;

[CustomEditor((typeof(LevelGenerator)))]
[CanEditMultipleObjects]
public class LevelGeneratorEditor:Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        var so = serializedObject;
        so.Update();

        if (GUILayout.Button("Generate"))
        {
            Generate();
        }
        
        if (GUILayout.Button("Clear Chunks"))
        {
            ClearChunks();
        }

    }

    private void Generate()
    {
        var levelGenerator = serializedObject.targetObject as LevelGenerator;
        if (levelGenerator)
        {
            levelGenerator.Generate();
        }
    }
    
    private void ClearChunks()
    {
        var levelGenerator = serializedObject.targetObject as LevelGenerator;
        if (levelGenerator)
        {
            levelGenerator.ClearChunks();
        }
    }
    
    
}
