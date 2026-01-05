
using Unity.MLAgents;

public interface IModularAgentComponent
{
    abstract void Initialize(EnvironmentParameters environmentParameters, ModularAgentProperties modularAgentProperties);

    abstract void InitializeAgentProperties();
    
    abstract void InitializeEnvironmentParameters();
    
    abstract void Reset();
}
