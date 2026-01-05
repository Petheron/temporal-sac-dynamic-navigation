using Unity.MLAgents;
using UnityEngine;

public class EuclideanReward:MonoBehaviour, IModularAgentComponent
{
    public Transform agentTransform;
    public Transform goalTransform;
    
    private int euclideanCounter;
    private float euclideanStepScaleFactor = 1.0f;
    private int euclideanStepLimit = (int) 1e6;
    private bool normalizeEucDistance;


    private Vector3 initialStartPosition;
    private Vector3 initialGoalPosition;
    private float initialEuclideanDistance = Mathf.Infinity;
    private float bestEuclideanDistance = Mathf.Infinity;
    
    private ModularAgentProperties modularAgentProp;
    private EnvironmentParameters envParams;
    
    public void Initialize(EnvironmentParameters environmentParameters, ModularAgentProperties modularAgentProperties)
    {
        modularAgentProp = modularAgentProperties;
        envParams = environmentParameters;
        InitializeAgentProperties();
        InitializeEnvironmentParameters();
    }

    public void InitializeAgentProperties()
    {
        if (modularAgentProp == null) return;
        euclideanStepScaleFactor = modularAgentProp.EUCLIDEAN_REWARD_SCALE_FACTOR;
        euclideanStepLimit = modularAgentProp.EUCLIDEAN_STEP_MAX;
        normalizeEucDistance = modularAgentProp.NORMALIZE_EUCLIDEAN_DISTANCE;
    }

    public void InitializeEnvironmentParameters()
    {
        if (envParams == null) return;
        euclideanStepScaleFactor = envParams.GetWithDefault("euclidean_step_scale_factor", euclideanStepScaleFactor);
        euclideanStepLimit = (int) envParams.GetWithDefault("euclidean_step_limit", euclideanStepLimit);
    }

    /// <summary>
    /// Calculate the euclidean reward for traversing a given distance. 
    /// </summary>
    /// <param name="distance"></param>
    /// <returns></returns>
    private float CalculateEuclideanDistanceReward(float distance)
    {
        var dist = distance;
        if (normalizeEucDistance)
        {
            dist /= initialEuclideanDistance;
        }
        return euclideanStepScaleFactor * dist;
    }

    private float GetEuclideanDistanceImprovement(Vector3 agentPosition, Vector3 goalPosition)
    {
        euclideanCounter++;
        float improvement = 0f;
        float currentDist = Vector3.Distance(agentPosition, goalPosition);
        if (currentDist < bestEuclideanDistance)
        {
            improvement = bestEuclideanDistance - currentDist;
            bestEuclideanDistance = currentDist;
            euclideanCounter = 0;
        }
        return Mathf.Max(improvement, 0f);
    }
    
    public float GetEuclideanReward()
    {
        if (!agentTransform || !goalTransform)
        {
            print($"agent: {!agentTransform}, goal: {!goalTransform}");
        }
        float euclideanImprovement = GetEuclideanDistanceImprovement(agentTransform.position, goalTransform.position);
        return CalculateEuclideanDistanceReward(euclideanImprovement);
    }
    

    /// <summary>
    /// Get the remaining euclidean reward (Sum of distance remaining to goal + stored euc reward from repetition).
    /// </summary>
    /// <returns></returns>
    public float GetRemainingEuclideanReward()
    {
        return CalculateEuclideanDistanceReward(bestEuclideanDistance);
    }
    
    public bool IsEuclideanTimeout()
    {
        if (euclideanStepLimit <= 0) return false;
        return euclideanCounter >= euclideanStepLimit;
    }
    
    public void Reset()
    {
        euclideanCounter = 0;
        
        initialStartPosition = agentTransform? agentTransform.position:Vector3.zero;
        initialGoalPosition = goalTransform? goalTransform.position:Vector3.zero;

        initialEuclideanDistance = Mathf.Max(Vector3.Distance(initialStartPosition, initialGoalPosition),Mathf.Epsilon);
        bestEuclideanDistance = initialEuclideanDistance;
        GetEuclideanDistanceImprovement(initialStartPosition, initialGoalPosition);
    }

    
}
