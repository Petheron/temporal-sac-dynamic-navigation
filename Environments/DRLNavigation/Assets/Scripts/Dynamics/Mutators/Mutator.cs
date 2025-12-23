public class Mutator
{
    private MutatorType type;
    private MutatorProperties properties;

    public MutatorType Type => type;
    public float AccelerationModifier => properties.accelerationModifier;
    public float DragModifier => properties.dragModifier;
    public float FrictionModifier => properties.frictionModifier;
    public float GravityModifier => properties.gravityModifier;
    public float AirControlModifier => properties.airControlModifier;

    public Mutator(MutatorType mutatorType)
    {
        type = mutatorType;
        properties = MutatorProperties.MutatorDictionary[type];
    }
    
    public Mutator(MutatorType mutatorType, MutatorProperties mutatorProperties)
    {
        type = mutatorType;
        properties = mutatorProperties;
    }
    
    public bool IsMutatorType(MutatorType newType)
    {
        return type == newType;
    }
}
