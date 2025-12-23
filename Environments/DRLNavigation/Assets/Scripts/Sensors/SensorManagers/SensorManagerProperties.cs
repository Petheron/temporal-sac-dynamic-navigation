public struct SensorManagerProperties
{
    public int ObservationStacks { get; private set; }
    public int ObservationSize { get; private set; }

    public SensorManagerProperties(int obsStack, int obsSize=0)
    {
        ObservationStacks = obsStack;
        ObservationSize = obsSize;
    }
}
