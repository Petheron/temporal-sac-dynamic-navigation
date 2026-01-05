using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

[CreateAssetMenu(fileName = "EnvironmentProperties", menuName = "Environment Properties", order = 0)]
public class EnvironmentProperties : ScriptableObject
{
    [Header("General Properties")]
    public int DEFAULT_SEED = 4298347;
    public float DEATH_FIELD_Y_OFFSET = -1.0f;
    
    [Header("Dynamic Properties")]
    public bool USE_DYNAMICS = false;

    [Header("Level Generalization Properties")]
    public int LEVEL_SEQUENCE_DELAY = 20000;
    public List<LevelType> LEVEL_SEQUENCE;

}