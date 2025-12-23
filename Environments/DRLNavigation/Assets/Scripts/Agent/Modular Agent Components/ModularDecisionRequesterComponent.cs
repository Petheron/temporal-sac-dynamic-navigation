
using System;
using Unity.MLAgents;
using UnityEngine;

public class ModularDecisionRequesterComponent:MonoBehaviour,IModularAgentComponent
{
    
    [Range(1, 20)] public int decisionPeriod;
    public bool takeActionsBetweenDecisions = true;
    
    protected EnvironmentParameters environmentParameters;
    protected AgentProperties agentProperties;
    private ModularNavAgent agent;
    private bool hasSubscribed;

    protected virtual void Awake()
    {
        agent = GetComponent<ModularNavAgent>();
    }

    public void Initialize(EnvironmentParameters environmentParameters, AgentProperties agentProperties)
    {
        this.environmentParameters = environmentParameters;
        this.agentProperties = agentProperties;
        InitializeAgentProperties();
        InitializeEnvironmentParameters();
    }

    public virtual void InitializeAgentProperties()
    {
        if (!agentProperties) return;
        
    }

    public virtual void InitializeEnvironmentParameters()
    {
        if (environmentParameters == null) return;
    }

    public void Reset()
    {
        return;
    }
    
    private void MakeRequest(int academyStepCount)
    {
        if (ShouldRequestDecision(academyStepCount))
        {
            // Note RequestDecision calls RequestAction itself, so no need to call it as well.
            agent.RequestDecision();
            return;
        }

        if (ShouldRequestAction(academyStepCount))
        {
            agent.RequestAction();
        }
    }

    // Should an agent request a decision?
    // Do NOT use agent.StepCount instead of academyStepCount, it messes with the resetting or terminal agents for some reason during training.
    private bool ShouldRequestDecision(int academyStepCount)
    {
        return academyStepCount % decisionPeriod == 0;
    }

    private bool ShouldRequestAction(int academyStepCount)
    {
        return takeActionsBetweenDecisions;
    }

    private void Subscribe()
    {
        if (hasSubscribed) return;
        Academy.Instance.AgentPreStep += MakeRequest;
        hasSubscribed = true;
    }
    
    // Unsubscribe from the academy 
    // Currently results in an error when closing the application as the Academy's FixedStepRunner singleton is not destroyed if this method is used in either OnDisable or OnDestroy.
    private void Unsubscribe()
    {
        if (!hasSubscribed) return;
        Academy.Instance.AgentPreStep -= MakeRequest;
        hasSubscribed = false;
    }

    private void OnEnable()
    {
        Subscribe();
    }
    
    private void OnDisable()
    {
        // Unsubscribe();
    }

    private void OnDestroy()
    {
        // Unsubscribe();
    }
}
