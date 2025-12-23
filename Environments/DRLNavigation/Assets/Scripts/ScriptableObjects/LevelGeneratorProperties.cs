using System;
using UnityEngine;

[CreateAssetMenu(fileName = "Level Generator Properties", menuName = "Level Generator Properties", order = 0)]
public class LevelGeneratorProperties : ScriptableObject
{
    [Header("Generation Properties")]
    public int LEVEL_GENERATION_DELAY = 20000;
    public int LEVEL_GENERATION_SIZE = 31;
    public int LEVEL_GENERATION_HEIGHT = 1;
    public bool USE_CONSTANT_LEVEL_GENERATION_HEIGHT = true;
    
    [Header("Seed Properties")]
    public int CHUNK_NUM_SEEDS = 8;
    public int CHUNK_INITIAL_SEED_INDEX = 0;
    
    [Header("Noise Properties")]
    public float NOISE_AMPLITUDE = 1f;
    public float NOISE_FREQUENCY = 0.05f;
    public float SEED_FREQUENCY = 0.5f;
    [Range(1, 10)] public int NOISE_ITERATIONS = 2;
    public float NOISE_AMPLITUDE_PERSISTENCE = 0.5f;
    
}
