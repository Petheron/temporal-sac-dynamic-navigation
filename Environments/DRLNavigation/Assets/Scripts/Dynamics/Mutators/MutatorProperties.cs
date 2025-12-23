using System.Collections.Generic;

public struct MutatorProperties
{
    public float accelerationModifier;
    public float dragModifier;
    public float frictionModifier;
    public float gravityModifier;
    public float airControlModifier;
    
    public MutatorProperties(float accelerationMod, float dragMod, float frictionMod, float gravMod, float airControlMod)
    {
        accelerationModifier = accelerationMod;
        dragModifier = dragMod;
        frictionModifier = frictionMod;
        gravityModifier = gravMod;
        airControlModifier = airControlMod;
    }
    

    public static Dictionary<MutatorType, MutatorProperties> MutatorDictionary = new()
    {
        {MutatorType.None, new MutatorProperties(1f, 1f, 1f, 1f, 1.0f)},
        {MutatorType.Water, new MutatorProperties(1f, 2f, 1f, 0.25f, 1.0f)},
        {MutatorType.Ice, new MutatorProperties(0.1f, 0.1f, 0.1f, 1f, 0.1f)},
        {MutatorType.Slow, new MutatorProperties(1f, 8f, 8f, 8f, 1.0f)},
        {MutatorType.LowGrav, new MutatorProperties(1f, 0f, 1f, 0.5f, 0.1f)},
        {MutatorType.HighGrav, new MutatorProperties(1f, 0f, 1f, 1.5f, 0.1f)},
        {MutatorType.Space, new MutatorProperties(1f, 0f, 0f, 0f, 0.1f)},
    };
}


