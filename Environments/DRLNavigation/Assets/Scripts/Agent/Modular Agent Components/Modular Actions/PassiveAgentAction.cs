using UnityEngine;

public class PassiveAgentAction:AgentAction
{
    public bool usePassiveAction;
    
    public virtual bool IsActive()
    {
        return false;
    }
}
