
using Unity.MLAgents;

public interface IModularAgentComponent
{
    abstract void Initialize(EnvironmentParameters environmentParameters, AgentProperties agentProperties);

    abstract void InitializeAgentProperties();
    
    abstract void InitializeEnvironmentParameters();
    
    abstract void Reset();
}
