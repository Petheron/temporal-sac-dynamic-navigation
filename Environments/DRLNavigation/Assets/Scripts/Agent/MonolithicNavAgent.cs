using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Random = UnityEngine.Random;
using Unity.MLAgents.Policies;

[RequireComponent(typeof(Health))]
[RequireComponent(typeof(HeuristicInputHandler))]

public class MonolithicNavAgent : Agent
{
    [HideInInspector] public int agentID;
    [HideInInspector] public int numAgents;
    [HideInInspector] public AgentType agentType;

    [Header("Agent Observations")]
    [SerializeField] private Vector3 vectorFromGoalToAgent;
    [SerializeField] private float agentDistanceFromGoal;
    [SerializeField] private float[] lastActions;
    private int numActions;
    [SerializeField] private bool observeRelPos = true;
    [SerializeField] private bool observeGlobalPos;
    [SerializeField] private bool observeLastActions = true;
    [SerializeField] private int numMovementObs;


    // Movement Properties
    private Health health;
    private BaseMovement movement;
    private HeuristicInputHandler heuristicInputHandler;
    private bool isHeuristic;
    private float heuristicMouseScale = 1f;
    private CapsuleCollider agentCollider;

    // Trajectory Recording parameters
    private List<Vector3> trajectoryAgentPos;
    private int trajectoryCounter;
    [SerializeField] private TrajectoryVisualizer trajectoryVisualizer;
    private TrajectoryPositionsRecorder trajectoryPositionsRecorder;
    private EndCondition endCondition;

    // Sensor managers
    private VectorSensorManager vectorSensorManager;
    private RayPerceptionBaseSensor3DManager rayPerceptionBaseSensor3DManager;
    private BoxcastSensorManager boxcastSensorManager;
    private DepthmapSensorManager depthmapSensorManager;
    private PositionalEncodingSensorManager[] positionalEncodingSensorManagers;

    private bool isRequestingDecision;

    [Header("Normalization Properties")]
    [SerializeField] private float normalizingGoalDistance = 40f;

    [Header("Goal Properties")]
    public Transform goalTransform;
    private Vector3 goalPosition;
    private bool useEdgeGoal;


    [Header("Environment Settings")] 
    public bool useRandomSeed;
    public int defaultSeed = 429834740;
    private int seed = -1;
    private int stepLimit;
    private float failurePenalty = -1f;
    private float timeStepPenalty = -1e-3f;
    private int actionRepeatRate = 1;
    
    private StatsRecorder statsRecorder;
    private EnvironmentParameters envParams;

    [Header("Euclidean Reward Values")]
    [SerializeField, ReadOnly] private int euclideanCounter;
    [Tooltip("Distance in which agent is considered to have reached the goal.")]
    private float epsilonDistance = 1.0f;
    private float euclideanStepScaleFactor = 1.0f;
    private int euclideanStepLimit = (int) 1e6;
    [SerializeField, ReadOnly] private float initialEuclideanDistance = Mathf.Infinity;
    [SerializeField, ReadOnly] private float bestEuclideanDistance = Mathf.Infinity;
    [SerializeField, ReadOnly] private float lastEuclideanImprovement;
    [SerializeField, ReadOnly] private float cumulativeEuclideanReturn;
    [SerializeField, ReadOnly] private float currentReturn;
    [SerializeField, ReadOnly] private float lastReturn;

    [Header("Goal Spawning Settings")]
    [SerializeField, ReadOnly] private float startingDistance;
    [SerializeField, ReadOnly] private float maxGoalDistance = 100f;

    [Header("Scriptable Objects")]
    public MonolithicAgentProperties monolithicAgentProperties;
    public VertexSamplingManager vertexSamplingManager;
    public AgentEvaluationManager agentEvaluationManager;

    [Header("Vertices")] 
    private Vertex startVertex;
    private Vertex goalVertex;
    private int startIndex;
    private int goalIndex;
    private int defaultStartIndex;
    
    private bool isContinueEvaluation=true;

    [Header("Testing")] 
    public bool drawGizmos = true;
    
    public event Action OnAgentPostInitialize;
    private bool isInitialized;
    private bool allowEnable;
    [HideInInspector] public CameraController cameraController;
    [SerializeField] private Camera visionCam;
    [HideInInspector] public AgentSpawner agentSpawner;
    private bool allowReset = true;

    private void Awake()
    {
        health = GetComponent<Health>();
        movement = GetComponent<BaseMovement>();
        heuristicInputHandler = GetComponent<HeuristicInputHandler>();

        vectorSensorManager = GetComponent<VectorSensorManager>();
        rayPerceptionBaseSensor3DManager = GetComponent<RayPerceptionBaseSensor3DManager>();
        boxcastSensorManager = GetComponent<BoxcastSensorManager>();
        positionalEncodingSensorManagers = GetComponents<PositionalEncodingSensorManager>();
        depthmapSensorManager = GetComponent<DepthmapSensorManager>();
        
        agentCollider ??= GetComponent<CapsuleCollider>(); 
        
        Academy.Instance.AgentPreStep += MakeRequest;
        
        InitializeAgentComponents();
    }

    public void InitializeAgentComponents()
    {
        if (OnAgentPostInitialize?.GetInvocationList()==null) return;
        if (isInitialized) return;
        isInitialized = true;
        OnAgentPostInitialize?.Invoke();
    }

    /// <summary>
    /// Request a Decision/Action as required based on current step count.
    /// </summary>
    /// <param name="academyStepCount"></param>
    private void MakeRequest(int academyStepCount)
    {
        if (!allowEnable) return;
        
        isRequestingDecision = IsDeciding(academyStepCount);
        if (isRequestingDecision)
        {
            // Note: requesting a decision also requests an action.
            RequestDecision();
        }
        else
        {
            RequestAction();
        }
    }

    /// <summary>
    /// Is the agent going to request a decision for this environment step?
    /// </summary>
    /// <param name="stepCount">Current agent step count</param>
    /// <returns></returns>
    private bool IsDeciding(int stepCount)
    {
        return stepCount % actionRepeatRate == 0;
    }

    private void SetSeed(int newSeed)
    {
        if (seed == newSeed) return;
        seed = newSeed;
        Random.InitState(seed);
    }

    /// <summary>
    /// Initialize the agent before episode begins.
    /// Use for setting properties via EnvironmentParameters and setting up observation sensors.
    /// </summary>
    public override void Initialize(){
        envParams = Academy.Instance.EnvironmentParameters;
        statsRecorder = Academy.Instance.StatsRecorder;

        actionRepeatRate = (int) envParams.GetWithDefault("action_repetition", monolithicAgentProperties.ACTION_REPETITION);

        var behaviourParams = GetComponent<BehaviorParameters>();
        numActions = behaviourParams.BrainParameters.ActionSpec.NumContinuousActions;
        lastActions = new float[numActions];
        
        if (movement)
        {
            movement.ReceiveEnvironmentParameters(envParams);
            movement.actionRepeatRate = actionRepeatRate;
            numMovementObs = movement.SetNumObs();
            
            movement.SetDefaultMovementProperties(monolithicAgentProperties);
            heuristicInputHandler.UpdateActionIndices(movement.usePitch);
        }
        
        InitializeObservationsToCollect();
        
        InitializeSensors();

        trajectoryVisualizer = Academy.Instance.IsCommunicatorOn ? null : FindObjectOfType<TrajectoryVisualizer>();
        trajectoryPositionsRecorder = FindObjectOfType<TrajectoryPositionManager>().TrajectoryPositionsRecorder;
        trajectoryCounter = 0;
    }
    
    public override void OnEpisodeBegin(){
        Reset();
    }

    /// <summary>
    /// Initialize the sensors of the agent via their sensor managers. 
    /// </summary>
    private void InitializeSensors()
    {
        var obsStacks = monolithicAgentProperties.OBS_STACKS;
        
        if (vectorSensorManager)
        {
            vectorSensorManager.observationSize = GetObsSize();
            vectorSensorManager.observationStacks = obsStacks;
            vectorSensorManager.SetProperties();
        }

        if (rayPerceptionBaseSensor3DManager)
        {
            rayPerceptionBaseSensor3DManager.observationStacks = obsStacks;
            rayPerceptionBaseSensor3DManager.SetProperties();
        }
        
        if (depthmapSensorManager)
        {
            depthmapSensorManager.observationStacks = obsStacks;
            depthmapSensorManager.SetProperties();
        }

        if (boxcastSensorManager)
        {
            boxcastSensorManager.observationStacks = obsStacks;
            boxcastSensorManager.SetProperties();
        }
        
        if (positionalEncodingSensorManagers.Length > 0)
        {
            foreach (var positionalEncodingSensorManager in positionalEncodingSensorManagers)
            {
                positionalEncodingSensorManager.observationStacks = obsStacks;
                positionalEncodingSensorManager.SetProperties();
            }
        }
    }

    /// <summary>
    /// Initialize environment for the beginning of a new episode.
    /// </summary>
    public void Reset(){
        
        // If not regenerating or changing level, prevent multiple resets before first observation has been made.
        if (Academy.Instance.AutomaticSteppingEnabled)
        {
            if (!allowReset) return;
            allowReset = false;
        }
        
        int newSeed = (int)envParams.GetWithDefault("seed",defaultSeed);
        if (useRandomSeed) newSeed = Random.Range(0, (int) Mathf.Pow(2, 16));
        SetSeed(newSeed);

        InitializeEnvironmentParameters();

        currentReturn = 0f;

        SampleVertices();
        ResetAgent();
        ResetGoalPosition();
        ResetEuclideanDistance();
        
        // reset trajectory list at initial position (prevent error on env.reset())
        trajectoryAgentPos = new List<Vector3> {transform.position};
    }

    private void InitializeEnvironmentParameters()
    {
        stepLimit = (int)envParams.GetWithDefault("step_limit",monolithicAgentProperties.DEFAULT_STEP_LIMIT);
        failurePenalty = envParams.GetWithDefault("failure_penalty",monolithicAgentProperties.FAIL_PENALTY);
        timeStepPenalty = envParams.GetWithDefault("time_step_penalty",monolithicAgentProperties.TIME_STEP_PENALTY);

        euclideanStepLimit = (int) envParams.GetWithDefault("euclidean_step_limit", monolithicAgentProperties.EUCLIDEAN_STEP_MAX);
        euclideanStepScaleFactor = envParams.GetWithDefault("euclidean_step_scale_factor", monolithicAgentProperties.EUCLIDEAN_REWARD_SCALE_FACTOR);
        epsilonDistance = monolithicAgentProperties.EPSILON_DISTANCE;
        
        maxGoalDistance = envParams.GetWithDefault("max_goal_distance",monolithicAgentProperties.MAX_GOAL_DISTANCE);
        normalizingGoalDistance = envParams.GetWithDefault("normalizing_distance", monolithicAgentProperties.NORMALIZATION_DISTANCE);
        
        useEdgeGoal = envParams.GetWithDefault("use_edge_goal", monolithicAgentProperties.SAMPLE_GOAL_VERTEX_FROM_START_VERTEX ? 1.0f : 0f) > 0f;
    }

    private void InitializeObservationsToCollect()
    {
        observeRelPos = envParams.GetWithDefault("observe_rel_pos", observeRelPos ? 1 : 0) > 0;
        observeGlobalPos = envParams.GetWithDefault("observe_global_pos", observeGlobalPos ? 1 : 0) > 0;
        observeLastActions = envParams.GetWithDefault("observe_last_actions", observeLastActions ? 1 : 0) > 0;
    }

    private int GetObsSize()
    {
        int size = 0;
        if (observeRelPos) size += 4;
        if (observeGlobalPos) size += 6;
        size += numMovementObs;
        if (observeLastActions) size += numActions;
        
        return size;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Collect normalized movement-dependent observations
        var movementObservations = movement.CollectVectorObservations();
        
        var agentPosition = movement.Position;
        if (goalTransform)
        {
            goalPosition = goalTransform.position;
            vectorFromGoalToAgent = agentPosition - goalPosition;
        }
        agentDistanceFromGoal = vectorFromGoalToAgent.sqrMagnitude;
        
        // normalize vector to goal and the distance away from goal.
        vectorFromGoalToAgent /= normalizingGoalDistance;
        agentDistanceFromGoal /= normalizingGoalDistance * normalizingGoalDistance;
        
        var actions = GetStoredActionBuffers().ContinuousActions;
        for (int i = 0; i < actions.Length; i++)
        {
            lastActions[i] = Mathf.Clamp(actions[i], -1f, 1f);
        }

        if (vectorSensorManager)
        {
            vectorSensorManager.ResetObsCount();
            if (observeRelPos)
            {
                vectorSensorManager.AddObservation(vectorFromGoalToAgent);    // (3)                            
                vectorSensorManager.AddObservation(agentDistanceFromGoal);    // (1)   
            }

            if (observeGlobalPos)
            {
                vectorSensorManager.AddObservation(agentPosition/normalizingGoalDistance);  // (3)
                vectorSensorManager.AddObservation(goalPosition/normalizingGoalDistance);   // (3)
            }

            vectorSensorManager.AddObservation(movementObservations);     // (9, 11, 14, 16, 17)
            
            if (observeLastActions)
            {
                vectorSensorManager.AddObservation(lastActions);              // (4+)
            } 

        }

        if (positionalEncodingSensorManagers.Length > 0)
        {
            foreach (var manager in positionalEncodingSensorManagers)
            {
                manager.GetOccupancyMapPositionObservation();
            }
        }
        
        // append the current position to trajectory
        trajectoryAgentPos?.Add(transform.position);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers){
        // This agent only take continuous actions. Booleans are discretized on the Unity side.
        var conActions = actionBuffers.ContinuousActions;
        movement.ProcessActions(conActions, heuristicMouseScale);

        float reward = RewardShaping();
        currentReturn += reward;
        SetReward(reward);

        if(IsTerminal()||IsTimeout())
        {
            lastReturn = currentReturn;
            UpdateCompletionRate();
            ProcessTrajectoryPositions();
            UpdateEdgeWeight(startVertex, goalVertex, CalculateEdgeWeight());
            if (IsTimeout())
            {
                EpisodeInterrupted();
            }
            else
            {
                EndEpisode();
            }
        }
    }
    
    /// <summary>
    /// Calculates the edge weight of a trajectory.
    ///
    /// W = s*stepCount/StepLimit + (1-s)
    /// 
    /// Weight values are between [0, 1], with non-successful trajectories being weighted at 1.0f. 
    /// </summary>
    /// <returns></returns>
    private float CalculateEdgeWeight()
    {
        Debug.Assert(MaxStep>0f);
        float stepRatio = stepLimit > 0 ? (float) (StepCount) / stepLimit : 0f;
        float s = IsAgentAtGoal() ? 1.0f : 0f;
        float weight = s * stepRatio + (1f - s);
        // print($"stepcount:{StepCount}, stepLimit: {m_StepLimit}, weight: {weight}");
        return weight;
    }

    private void UpdateEdgeWeight(Vertex start, Vertex goal, float weight)
    {
        vertexSamplingManager.UpdateEdgeWeight(start,goal, weight);
    }

    /// <summary>
    /// Update the completion statistics for export to python via StatsRecorder.
    /// </summary>
    private void UpdateCompletionRate()
    {
        var didAgentSucceed = 0f;
        var didAgentFail = 0f;
        var didEuclideanFail = 0f;
        var didAgentTimeout = 0f;
        
        if (IsAgentAtGoal())
        {
            didAgentSucceed = 1f;
            endCondition = EndCondition.Success;
        }
        else if(IsAgentDead())
        {
            didAgentFail = 1f;
            endCondition = EndCondition.Failure;
        }
        else if (IsEuclideanTimeout())
        {
            didEuclideanFail = 1f;
            endCondition = EndCondition.EucFail;
        }
        else if(IsTimeout())
        {
            didAgentTimeout = 1f;
            endCondition = EndCondition.Timeout;
        }

        statsRecorder.Add("success_rate",didAgentSucceed);
        statsRecorder.Add("fail_rate",didAgentFail);
        statsRecorder.Add("euclidean_fail_rate", didEuclideanFail);
        statsRecorder.Add("timeout_fail_rate",didAgentTimeout);
    }

    public void Interrupt()
    {
        statsRecorder.Add("timeout_fail_rate",1);
        EpisodeInterrupted();
    }

    private void ProcessTrajectoryPositions()
    {
        trajectoryCounter++;
        int episodeID = agentID + trajectoryCounter * numAgents;
        trajectoryPositionsRecorder?.Add(episodeID, trajectoryAgentPos, endCondition, goalPosition);
        if(trajectoryVisualizer) trajectoryVisualizer.StorePositions(episodeID,trajectoryAgentPos,endCondition,goalPosition, movement.complexActionPositions, agentType, TrajectoryType.DRL, startIndex, goalIndex);
        trajectoryAgentPos.Clear();
    }

    private float RewardShaping()
    {
        float reward = 0f;
        
        reward += timeStepPenalty*actionRepeatRate;

        reward += GetEuclideanImprovementReward();
        
        if (IsAgentAtGoal())
        {
            reward += CalculateGoalReward();
        }

        if (IsAgentDead())
        {
            reward += failurePenalty;
        }

        if (IsEuclideanTimeout())
        {
            reward += monolithicAgentProperties.EUCLIDEAN_TIMEOUT_PENALTY;
        }
        
        return reward;
    }

    /// <summary>
    /// Get the euclidean improvement reward for the last step if possible.
    /// Returns 0 if no improvement to distance made or currently repeating an action.
    /// </summary>
    /// <returns></returns>
    private float GetEuclideanImprovementReward()
    {
        float euclideanImprovement = UpdateEuclideanDistance();
        float euclideanReward = CalculateEuclideanDistanceReward(euclideanImprovement);

        lastEuclideanImprovement = euclideanReward > 0 ? euclideanReward : lastEuclideanImprovement;
        cumulativeEuclideanReturn += euclideanReward;
        
        // Goal is to sync euc reward with decision step to prevent reduction in stable return.
        float outputReward = 0f;
        
        if (Academy.Instance.StepCount % actionRepeatRate==0)
        {
            outputReward = GetCumulativeEuclideanReward();
        }

        return outputReward;
    }

    /// <summary>
    /// Obtain the accumulated euclidean reward over repeated actions on either a new decision or terminal state.
    /// </summary>
    /// <returns></returns>
    private float GetCumulativeEuclideanReward()
    {
        var outputReward = cumulativeEuclideanReturn;
        cumulativeEuclideanReturn = 0;
        return outputReward;
    }

    /// <summary>
    /// Returns how much euclidean reward the agent should receive.
    /// </summary>
    /// <param name="distance">The distance of euclidean improvement</param>
    /// <returns>The calculated euclidean reward.</returns>
    private float CalculateEuclideanDistanceReward(float distance)
    {
        var dist = distance;
        if (monolithicAgentProperties.NORMALIZE_EUCLIDEAN_DISTANCE)
        {
            dist /= initialEuclideanDistance;
        }

        return euclideanStepScaleFactor * dist;
    }

    /// <summary>
    /// Returns the goal reward. This combines the goal signal,
    /// the remaining euclidean distance, and any cumulative euclidean reward collected as a result of action repetition.
    /// </summary>
    /// <returns></returns>
    private float CalculateGoalReward()
    {
        float reward = MathF.Abs(monolithicAgentProperties.GOAL_REWARD);
        reward += CalculateEuclideanDistanceReward(bestEuclideanDistance);
        reward += GetCumulativeEuclideanReward();
        return reward;
    }

    /// <summary>
    /// Did the agent reach the goal position?
    /// </summary>
    /// <returns></returns>
    private bool IsAgentAtGoal()
    {
        if (!goalTransform) return false;
        return Vector3.Distance(movement.baseTransform.position, goalTransform.position) <= epsilonDistance;
    }

    /// <summary>
    /// Did the agent fail to improve its best euclidean distance?
    /// </summary>
    /// <returns></returns>
    private bool IsEuclideanTimeout()
    {
        if (euclideanStepLimit <= 0) return false;
        return euclideanCounter >= euclideanStepLimit;
    }

    /// <summary>
    /// Did the agent fall off the map?
    /// </summary>
    /// <returns></returns>
    private bool IsAgentDead()
    {
        // return movement.baseTransform.position.y <= agentProp.DEATH_Y_POSITION;
        return !health.IsAlive;
    }
    
    /// <summary>
    /// Has the agent reached a terminal state? Did the agent successfully reach the goal or fall off the map?
    /// </summary>
    /// <returns></returns>
    private bool IsTerminal(){
        return IsAgentAtGoal() || IsEuclideanTimeout() || IsAgentDead();
    }

    /// <summary>
    /// Was the episode interrupted? Was the stepLimit reached?
    /// </summary>
    /// <returns></returns>
    private bool IsTimeout()
    {
        if (stepLimit <= 0) return false;
        return StepCount >= stepLimit;
    }

    /// <summary>
    /// Samples the Start and Goal vertices. Handles edge evaluation, random start positions, and edge goals.
    /// </summary>
    private void SampleVertices()
    {
        if (agentEvaluationManager)
        {
            if (agentEvaluationManager.performEvaluation)
            {
                if (isContinueEvaluation)
                {
                    isContinueEvaluation = agentEvaluationManager.SampleEvaluationNodes(agentType, agentID, ref startVertex, ref goalVertex, ref startIndex, ref goalIndex);
                    return;
                }
                Destroy(transform.parent.gameObject);
            }
        }

        // If not evaluating, either select at random or from a given starting index
        startVertex = monolithicAgentProperties.IS_AGENT_POSITION_RANDOM
            ? vertexSamplingManager.SampleVertex(maxGoalDistance, Vector3.positiveInfinity)
            : vertexSamplingManager.GetVertex(defaultStartIndex);
        var startPosition = startVertex.GetPosition();
        
        goalVertex = useEdgeGoal? 
            vertexSamplingManager.SampleEdgeVertex(startVertex) :
            vertexSamplingManager.SampleVertex(maxGoalDistance, startPosition, monolithicAgentProperties.EPSILON_DISTANCE);
        
        startIndex = vertexSamplingManager.GetVertexIndex(startVertex);
        goalIndex = vertexSamplingManager.GetVertexIndex(goalVertex);
    }

    /// <summary>
    /// Reset Agent parameters upon new episode.
    /// </summary>
    private void ResetAgent(){
        
        // set initial position & rotation
        Vector3 posOffset = (agentCollider ? agentCollider.height/2.0f:1.0f) * Vector3.up;
        Vector3 pos = startVertex.GetPosition() + posOffset;

        Quaternion rot = 
            monolithicAgentProperties.IS_AGENT_ROTATION_RANDOM? 
            Quaternion.Euler(0f, Random.value * 360f, 0f):
            Quaternion.identity;

        movement.ResetMovement(pos, rot);
        health.Reset();
        
        // clear last action array
        if (lastActions != null)
        {
            Array.Clear(lastActions,0,lastActions.Length);
        }
        
        
    }

    /// <summary>
    /// Set goal position upon new episode.
    /// </summary>
    private void ResetGoalPosition()
    {
        if (!goalVertex) return;
        if (!goalTransform) return;
        goalTransform.position = goalVertex.GetPosition();
        goalTransform.SetParent(goalVertex.GetParent());
    }
    
    /// <summary>
    /// Reset euclidean parameters upon new episode.
    /// </summary>
    private void ResetEuclideanDistance()
    {
        
        startingDistance = goalTransform ? Mathf.Max(Vector3.Distance(movement.baseTransform.position, goalTransform.position), Mathf.Epsilon) : Mathf.Epsilon;
        initialEuclideanDistance = startingDistance;
        bestEuclideanDistance = initialEuclideanDistance;
        lastEuclideanImprovement = 0f;
        cumulativeEuclideanReturn = 0f;
        euclideanCounter = 0;

        UpdateEuclideanDistance();
    }

    /// <summary>
    /// Update the best euclidean distance.
    /// </summary>
    /// <returns>Improvement of the best euclidean distance.</returns>
    private float UpdateEuclideanDistance()
    {
        if (!goalTransform) return 0f;
        euclideanCounter++;
        float improvement = 0f;
        float currentDist = Vector3.Distance(movement.baseTransform.position, goalTransform.position);
        if (currentDist < bestEuclideanDistance)
        {
            improvement = bestEuclideanDistance - currentDist;
            bestEuclideanDistance = currentDist;
            euclideanCounter = 0;
        }
        return Mathf.Max(improvement, 0f);
    }

    public override void Heuristic(in ActionBuffers actionsOut){
        if (!isHeuristic)
        {
            isHeuristic = true;
            heuristicMouseScale = heuristicInputHandler.maximumMouseDelta;
        }
        var continuousActionsOut = actionsOut.ContinuousActions;
        heuristicInputHandler.HeuristicInputsToActions(continuousActionsOut);
    }
    
    
    public void AllowEnable()
    {
        // Toggle enabled state of gameobject to force OnEnable() to be called.
        if(enabled) enabled = false;
        allowEnable = true;
        enabled = true;
        
    }

    protected override void OnEnable()
    {
        if(!allowEnable) return;
        base.OnEnable();
    }

    protected override void OnDisable()
    {
        if(!allowEnable) return;
        RemoveCamera();
        UnsubscribeToEvents();
        base.OnDisable();
    }

    public void SetCameraController(CameraController camCon)
    {
        if (!camCon) return;
        cameraController = camCon;
        if(!visionCam) visionCam = GetComponentInChildren<Camera>();
        if (!visionCam) return;
        cameraController.AddCamera(visionCam);
    }

    private void RemoveCamera()
    {
        if (!visionCam) return;
        if (!cameraController) return;
        cameraController.RemoveCamera(visionCam);
    }

    private void UnsubscribeToEvents()
    {
        OnAgentPostInitialize = null;
        agentSpawner.OnAcademySteppingEnabled -= AllowEnable;
    }
    
    private void OnDestroy()
    {
        if (agentEvaluationManager)
        {
            agentEvaluationManager.TryInvokeOnEvalComplete();
        }
    }


#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        // if (!drawGizmos) return;
        Gizmos.color = Color.white;
        Gizmos.DrawRay(goalPosition, vectorFromGoalToAgent*normalizingGoalDistance);
        Gizmos.color = Color.magenta;
        // Gizmos.DrawWireSphere(Vector3.zero,m_MaxGoalDistance);
    }
#endif
}
