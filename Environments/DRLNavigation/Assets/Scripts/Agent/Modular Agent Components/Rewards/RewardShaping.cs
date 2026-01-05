using System;
using Unity.MLAgents;
using UnityEngine;

public class RewardShaping:MonoBehaviour, IModularAgentComponent
{
    private float episodeReturn;
    
    private ModularAgentProperties modularAgentProp;
    private EnvironmentParameters envParams;

    // Sparse Rewards
    private float successReward;
    private float failurePenalty;
    private float timeoutPenalty;
    private float euclideanTimoutPenalty;

    // Dense Rewards 
    private float timeStepPenalty;
    [SerializeField] private EuclideanReward euclideanReward;
    
    // Testing
    [ReadOnly, SerializeField] private float currentReturn;
    [ReadOnly, SerializeField] private float lastEpisodeReturn;

    public void Initialize(EnvironmentParameters environmentParameters, ModularAgentProperties modularAgentProperties)
    {
        modularAgentProp = modularAgentProperties;
        envParams = environmentParameters;
        InitializeAgentProperties();
        InitializeEnvironmentParameters();
        if(euclideanReward) euclideanReward.Initialize(envParams, modularAgentProp);
    }

    public void InitializeAgentProperties()
    {
        if (modularAgentProp == null) return;
        
        successReward = modularAgentProp.GOAL_REWARD;
        failurePenalty = modularAgentProp.FAIL_PENALTY;
        timeoutPenalty = modularAgentProp.TIMEOUT_PENALTY;
        euclideanTimoutPenalty = modularAgentProp.EUCLIDEAN_TIMEOUT_PENALTY;
        
        timeStepPenalty = modularAgentProp.TIME_STEP_PENALTY;
    }

    public void InitializeEnvironmentParameters()
    {
        if (envParams == null) return;
        
        successReward = envParams.GetWithDefault("success_reward", successReward);
        failurePenalty = envParams.GetWithDefault("failure_penalty", failurePenalty);
        timeStepPenalty = envParams.GetWithDefault("time_step_penalty", timeStepPenalty);
    }

    private float GetSparseReward(EndCondition endCondition)
    {
        float reward = 0f;
        switch (endCondition)
        {
            case EndCondition.Success: 
                reward += successReward;
                if(euclideanReward) reward += euclideanReward.GetRemainingEuclideanReward();
                break;
            case EndCondition.Failure: 
                reward += failurePenalty;
                break;
            case EndCondition.EucFail: 
                reward+= euclideanTimoutPenalty;
                break;
            case EndCondition.Timeout:
                reward += timeoutPenalty;
                break;
            default:
                break;
        }
        currentReturn += reward;
        return reward;
    }

    private float GetDenseReward()
    {
        float r = 0f;
        r += timeStepPenalty;
        if (euclideanReward)
        {
            r += euclideanReward.GetEuclideanReward();
        }
        currentReturn += r;
        return r;
    }

    public float GetReward(EndCondition endCondition)
    {
        return GetDenseReward() + GetSparseReward(endCondition);
    }

    public void Reset()
    {
        lastEpisodeReturn = currentReturn;
        currentReturn = 0f;
        if(euclideanReward) euclideanReward.Reset();
    }


    public bool IsEuclideanTimeout()
    {
        return euclideanReward && euclideanReward.IsEuclideanTimeout();
    }
}
