using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Random = UnityEngine.Random;
using Unity.MLAgents.Policies;
using UnityEngine.Serialization;

[RequireComponent(typeof(Health))]
[RequireComponent(typeof(PlayerActionsManager))]
[RequireComponent(typeof(ModularMovement))]

public class ModularNavAgent : Agent, IModularAgentComponent
{
    // public variables
    [ReadOnly] public int agentID;
    [HideInInspector] public int numAgents;
    [HideInInspector] public AgentType agentType;
    
    [Header("Transforms")]
    [SerializeField] private Transform goalTransform;
    
    [Header("Scriptable Objects")]
    [SerializeField] private ModularAgentProperties modularAgentProperties;
    [SerializeField] private VertexSamplingManager vertexSamplingManager;
    public AgentEvaluationManager agentEvaluationManager;
    
    // local variables
    private float epsilonDistance = 1f;
    private int stepLimit;
    private EndCondition endCondition;
    private float normalizationDistance = 40f;
    private int numActions;

    private bool allowEnable;
    
    private Vector3 agentStartPosition;
    private Vector3 goalStartPosition;
    private Quaternion agentStartRotation;

    private VectorNormalizationType vectorNormalizationType;
    private VectorSpatialOrientationType vectorSpatialOrientationType;
    
    // vertices variables
    private Vertex startVertex;
    private Vertex goalVertex;
    private int startIndex;
    private int goalIndex;
    private bool isContinueEvaluation=true;
    private float maxGoalDistance;
    private bool useEdgeGoal;
    
    // Trajectory Recording parameters
    private List<Vector3> trajectoryAgentPos;
    private int trajectoryCounter;
    private TrajectoryVisualizer trajectoryVisualizer;
    private TrajectoryPositionsRecorder trajectoryPositionsRecorder;
    [HideInInspector] public CameraController cameraController;
    
    // sensor managers
    private VectorSensorManager vectorSensorManager;
    private RayPerceptionBaseSensor3DManager rayPerceptionBaseSensor3DManager;
    private BoxcastSensorManager boxcastSensorManager;
    private DepthmapSensorManager depthmapSensorManager;
    private PositionalEncodingSensorManager[] positionalEncodingSensorManagers;
    
    // Serialized local variables
    [Header("Observation Toggles")]
    [SerializeField] private bool observeRelativePositions = true;
    [SerializeField] private bool observeGlobalPositions;
    [SerializeField] private bool observeLastActions;

    [Header("Observations")]
    [SerializeField, ReadOnly] private Vector3 goalDirection;
    [SerializeField, ReadOnly] private float goalDistance;
    [SerializeField, ReadOnly] private List<float> observations;
    [SerializeField, ReadOnly] private List<float> lastActions;

    // Components
    private PlayerActionsManager playerActionsManager;
    private ModularMovement movement;
    private Health health;
    private RewardShaping rewardShaping;
    private ActionRepeater actionRepeater;
    private BehaviorParameters behaviorParameters;
    private CapsuleCollider agentCollider;
    [SerializeField] private Camera visionCam;

    // singletons
    private StatsRecorder statsRecorder;
    private EnvironmentParameters envParams;
    public AgentSpawner agentSpawner;
    
    // variables
    private bool IsAgentAtGoal => Vector3.Distance(AgentPosition, GoalPosition) <= epsilonDistance;
    private bool IsAgentDead => !health.IsAlive;
    private bool IsTimeOut => StepCount >= stepLimit && stepLimit>0;
    private bool IsEucFail => rewardShaping.IsEuclideanTimeout();
    private Vector3 AgentPosition => transform.position - agentCollider.height / 2f * transform.up;
    private Vector3 GoalPosition
    {
        get => goalTransform ? goalTransform.position:Vector3.zero;
        set => goalTransform.position = value;
    }

    // prevent double reset common when evaluating.
    private bool allowReset = true;

    // Action Events
    public event Action OnReset;
    public event Action<EnvironmentParameters, ModularAgentProperties> OnAgentPreInitialize;
    public event Action OnAgentPostInitialize;

    [SerializeField, ReadOnly] private bool isInitialized;

    // Called on class construction (always)
    private void Awake()
    {
        envParams = Academy.Instance.EnvironmentParameters;
        statsRecorder = Academy.Instance.StatsRecorder;
        
        behaviorParameters = GetComponent<BehaviorParameters>();
        health = GetComponent<Health>();
        playerActionsManager = GetComponent<PlayerActionsManager>();
        movement = GetComponent<ModularMovement>();
        actionRepeater = GetComponent<ActionRepeater>();
        rewardShaping = GetComponent<RewardShaping>();
        agentCollider = GetComponent<CapsuleCollider>();
        
        vectorSensorManager = GetComponent<VectorSensorManager>();
        rayPerceptionBaseSensor3DManager = GetComponent<RayPerceptionBaseSensor3DManager>();
        boxcastSensorManager = GetComponent<BoxcastSensorManager>();
        positionalEncodingSensorManagers = GetComponents<PositionalEncodingSensorManager>();
        depthmapSensorManager = GetComponent<DepthmapSensorManager>();
        
        SubscribeToEvents();
        OnAgentPreInitialize?.Invoke(envParams,modularAgentProperties);
        InitializeAgentComponents();
    }

    // Call after the Academy has been reset via Dispose().
    public void InitializeAgentComponents()
    {
        if (OnAgentPostInitialize?.GetInvocationList()==null) return;
        if (isInitialized) return;
        isInitialized = true;
        OnAgentPostInitialize?.Invoke();
    }

    public void Initialize(EnvironmentParameters environmentParameters, ModularAgentProperties modularAgentProperties)
    {
        InitializeAgentProperties();
        InitializeEnvironmentParameters();
    }

    public override void Initialize()
    {
        if(!trajectoryVisualizer && !Academy.Instance.IsCommunicatorOn) trajectoryVisualizer = FindObjectOfType<TrajectoryVisualizer>();
        trajectoryPositionsRecorder ??= FindObjectOfType<TrajectoryPositionManager>()?.TrajectoryPositionsRecorder;
        trajectoryCounter = 0;
        
        agentStartPosition = AgentPosition;
        goalStartPosition = GoalPosition;
        
        InitializeSensors();
    }

    public override void OnEpisodeBegin()
    {
        // Debug.LogWarning($"Episode Beginning on Step: {Academy.Instance.StepCount}");
        Reset();
    }

    // Only called before decision request / once per N action repetition steps
    public override void CollectObservations(VectorSensor sensor)
    {
        // allow for a reset after first observation made
        if (!allowReset) allowReset = true;
        
        // If start/goal vertices are null (due to regeneration from PCG or level sequence spawn), interrupt episode and get new vertices.
        // Shouldn't occur as agent is interrupted before AND after level change.
        if (!startVertex || !goalVertex)
        {
            // Debug.LogWarning($"Collect Obs: At {Academy.Instance.StepCount}: Is Null? Start: {!startVertex}, Goal: {!goalVertex}");
            Reset();
        }
        
        observations = new List<float>();

        goalDirection = GoalPosition - AgentPosition;
        goalDistance = goalDirection.magnitude;

        if (vectorSensorManager)
        {
            vectorSensorManager.ResetObsCount();
            
            if (observeRelativePositions)
            {
                var goalVector = vectorSpatialOrientationType == VectorSpatialOrientationType.LocalSpace
                    ? movement.WorldToLocal.MultiplyVector(goalDirection)
                    : goalDirection;
                observations.AddRange(AgentUtility.GetNormalizedVectorObservation(goalVector, normalizationDistance, vectorNormalizationType));
            }
            
            if (observeGlobalPositions)
            {
                observations.AddRange(AgentUtility.VectorToList(AgentPosition / normalizationDistance));
                observations.AddRange(AgentUtility.VectorToList(GoalPosition / normalizationDistance));
            }

            if(observeLastActions) observations.AddRange(lastActions);
            
            observations.AddRange(movement.CollectObservations());
            
            vectorSensorManager.AddObservation(observations);
        }
        
        if (positionalEncodingSensorManagers.Length > 0)
        {
            foreach (var manager in positionalEncodingSensorManagers)
            {
                manager.GetOccupancyMapPositionObservation();
            }
        }

        trajectoryAgentPos?.Add(transform.position);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var conActions = actions.ContinuousActions;
        movement.ProcessActions(conActions);
        StoreLastActions(conActions);
        
        DetermineEndCondition();
        CalculateReward();
        
        if (endCondition == EndCondition.Null) return;
        
        UpdateStatsRecorder();
        ProcessTrajectoryPositions();

        if (endCondition == EndCondition.Timeout)
        {
            EpisodeInterrupted();
            return;
        }
        EndEpisode();
        
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        playerActionsManager.HeuristicInputsToActions(actionsOut.ContinuousActions);
    }

    public void InitializeAgentProperties()
    {
        stepLimit = modularAgentProperties.DEFAULT_STEP_LIMIT;
        epsilonDistance = modularAgentProperties.EPSILON_DISTANCE;
        normalizationDistance = modularAgentProperties.NORMALIZATION_DISTANCE;
        maxGoalDistance = modularAgentProperties.MAX_GOAL_DISTANCE;
        useEdgeGoal = modularAgentProperties.SAMPLE_GOAL_VERTEX_FROM_START_VERTEX;

        vectorNormalizationType = modularAgentProperties.VECTOR_NORMALIZATION_TYPE;
        vectorSpatialOrientationType = modularAgentProperties.VECTOR_SPATIAL_ORIENTATION;
    }
    
    public void InitializeEnvironmentParameters()
    {
        stepLimit = (int)envParams.GetWithDefault("step_limit", stepLimit);
        normalizationDistance = envParams.GetWithDefault("normalizing_distance", modularAgentProperties.NORMALIZATION_DISTANCE);
        maxGoalDistance = envParams.GetWithDefault("max_goal_distance", maxGoalDistance);
        useEdgeGoal = envParams.GetWithDefault("use_edge_goal", useEdgeGoal ? 1.0f : 0f) > 0f;
        
        observeRelativePositions = envParams.GetWithDefault("observe_rel_pos", observeRelativePositions ? 1 : 0) > 0;
        observeGlobalPositions = envParams.GetWithDefault("observe_global_pos", observeGlobalPositions ? 1 : 0) > 0;
        observeLastActions = envParams.GetWithDefault("observe_last_actions", observeLastActions ? 1 : 0) > 0;

        vectorNormalizationType =
            (VectorNormalizationType) (int) envParams.GetWithDefault("vector_normalization_type", (int) vectorNormalizationType);
        vectorSpatialOrientationType =
            (VectorSpatialOrientationType) envParams.GetWithDefault("vector_spatial_orientation_type",
                (int) vectorSpatialOrientationType);
    }

    private void SetNumContinuousActions()
    {
        numActions = movement.actionTypes.Count;
        behaviorParameters.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(numActions);
    }
    
    private void StoreLastActions(ActionSegment<float> conActions)
    {
        lastActions = new List<float>(conActions.Array);
        for (int i = 0; i < lastActions.Count; i++)
        {
            lastActions[i] = Mathf.Clamp(lastActions[i], -1f, 1f);
        }
    }
    
    private void ProcessTrajectoryPositions()
    {
        int episodeID = agentID + trajectoryCounter * numAgents;
        trajectoryPositionsRecorder?.Add(episodeID, trajectoryAgentPos, endCondition, GoalPosition);
        if(trajectoryVisualizer) trajectoryVisualizer.StorePositions(episodeID,trajectoryAgentPos,endCondition,GoalPosition, movement.GetComplexActionPositions(), agentType, TrajectoryType.DRL, startIndex, goalIndex);
        trajectoryAgentPos.Clear();
        trajectoryCounter++;
    }

    private void CalculateReward()
    {
        AddReward(rewardShaping.GetReward(endCondition));
    }

    private void DetermineEndCondition()
    {
        endCondition = EndCondition.Null;
        
        if (IsAgentAtGoal) endCondition = EndCondition.Success;
        else if (IsAgentDead) endCondition = EndCondition.Failure;
        else if (IsTimeOut) endCondition = EndCondition.Timeout;
        else if (IsEucFail) endCondition = EndCondition.EucFail;
    }
    
    private void UpdateStatsRecorder()
    {
        if (endCondition == EndCondition.Null) return;
        statsRecorder.Add("success_rate",endCondition==EndCondition.Success?1f:0f);
        statsRecorder.Add("fail_rate",endCondition==EndCondition.Failure?1f:0f);
        statsRecorder.Add("euclidean_fail_rate", endCondition==EndCondition.EucFail?1f:0f);
        statsRecorder.Add("timeout_fail_rate",endCondition==EndCondition.Timeout?1f:0f);

        behaviorParameters.BehaviorType = BehaviorType.InferenceOnly;
    }
    
    private int CalculateVectorSensorObservationSize()
    {
        int size = 0;
        if (observeRelativePositions) size += 4;
        if (observeGlobalPositions) size += 6;
        if(movement) size += movement.GetNumMovementObservations();
        if (observeLastActions) size += numActions;
        
        return size;
    }
    
    /// <summary>
    /// Initialize the sensors of the agent via their sensor managers. 
    /// </summary>
    private void InitializeSensors()
    {
        
        SensorManagerProperties properties = new SensorManagerProperties(modularAgentProperties.OBS_STACKS, CalculateVectorSensorObservationSize());

        if(vectorSensorManager)vectorSensorManager.Initialize(properties);

        if (rayPerceptionBaseSensor3DManager) rayPerceptionBaseSensor3DManager.Initialize(properties);
        
        if (depthmapSensorManager) depthmapSensorManager.Initialize(properties);
        
        if(boxcastSensorManager) boxcastSensorManager.Initialize(properties);
        
        foreach (var positionalEncodingSensorManager in positionalEncodingSensorManagers)
        {
            positionalEncodingSensorManager.Initialize(properties);
        }
    }
    
    /// <summary>
    /// Samples the Start and Goal vertices. Handles edge evaluation, random start positions, and edge goals.
    /// </summary>
    private void SampleVertices()
    {
        if (!vertexSamplingManager) return;
        
        if (agentEvaluationManager)
        {
            if (agentEvaluationManager.performEvaluation)
            {
                isContinueEvaluation = agentEvaluationManager.SampleEvaluationNodes(agentType, agentID, ref startVertex, ref goalVertex, ref startIndex, ref goalIndex);
                if (isContinueEvaluation)
                {
                    return;
                }

                if (!Academy.Instance.AutomaticSteppingEnabled) return;
                
                RemoveCamera();
                if (agentSpawner)
                {
                    agentSpawner.RemoveAgent(gameObject.transform.parent.gameObject);
                }
                Destroy(transform.parent.gameObject);
                return;
            }
        }
        
        // If not evaluating, either select at random or from a given starting index
        startVertex = modularAgentProperties.IS_AGENT_POSITION_RANDOM
            ? vertexSamplingManager.SampleVertex(maxGoalDistance, Vector3.positiveInfinity)
            : vertexSamplingManager.GetVertex(0);
        
        goalVertex = useEdgeGoal? 
            vertexSamplingManager.SampleEdgeVertex(startVertex) :
            vertexSamplingManager.SampleVertex(maxGoalDistance, startVertex?startVertex.GetPosition():Vector3.zero, modularAgentProperties.EPSILON_DISTANCE);

        // Notify if vertices are null (should only occur during forced environment reset from python code, which immediately is replaced by a reset after the graph completes)
        // if (!startVertex || !goalVertex)
        // {
        //     Debug.LogWarning($"Sample Vertices: At Step {Academy.Instance.StepCount}: Is Null? Start: {!startVertex}, Goal: {!goalVertex}");
        // }
        
        startIndex = vertexSamplingManager.GetVertexIndex(startVertex);
        goalIndex = vertexSamplingManager.GetVertexIndex(goalVertex);
    }

    private void SetPositionsAndRotation()
    {
        if(startVertex) agentStartPosition = startVertex.GetPosition();
        agentStartRotation = modularAgentProperties.IS_AGENT_ROTATION_RANDOM
            ? Quaternion.Euler(0f, Random.Range(0f, 360f), 0f)
            : Quaternion.identity;

        movement.SetPositionAndRotation(agentStartPosition, agentStartRotation);

        if (goalVertex)
        {
            GoalPosition = goalVertex.GetPosition();
            // goalTransform.SetParent(goalVertex.GetParent());
        }
    }

    public void Reset()
    {
        // If not regenerating or changing level, prevent multiple resets before first observation has been made.
        if (Academy.Instance.AutomaticSteppingEnabled)
        {
            if (!allowReset) return;
            allowReset = false;
        }
        
        endCondition = EndCondition.Null;
        InitializeEnvironmentParameters();
        SampleVertices();
        
        OnReset?.Invoke();
        
        SetPositionsAndRotation();

        trajectoryAgentPos = new List<Vector3>();
        
        observations = new List<float>();
    }

    private void SubscribeToEvents()
    {
        OnReset += health.Reset;
        OnReset += movement.Reset;
        OnReset += rewardShaping.Reset;

        OnAgentPreInitialize += Initialize;
        OnAgentPreInitialize += movement.Initialize;
        OnAgentPreInitialize += rewardShaping.Initialize;
        OnAgentPreInitialize += actionRepeater.Initialize;

        movement.OnInitialized += SetNumContinuousActions;
    }
    
    private void UnsubscribeToEvents()
    {
        OnReset -= health.Reset;
        OnReset -= movement.Reset;
        OnReset -= rewardShaping.Reset;
        
        OnAgentPreInitialize -= Initialize;
        OnAgentPreInitialize -= movement.Initialize;
        OnAgentPreInitialize -= rewardShaping.Initialize;
        OnAgentPreInitialize -= actionRepeater.Initialize;
        
        movement.OnInitialized -= SetNumContinuousActions;

        OnAgentPostInitialize -= agentSpawner.IncrementInitializedAgentsCount;
        agentSpawner.OnAcademySteppingEnabled -= AllowEnable;
    }

    public void AllowEnable()
    {
        allowEnable = true;
        
        // Toggle enabled state of gameobject to force OnEnable() to be called.
        if(enabled) enabled = false;
        enabled = true;
        
        if (actionRepeater)
        {
            // Toggle actionrepeater enabled state to force OnEnable() to be called.
            if(actionRepeater.enabled) actionRepeater.enabled = false;
            actionRepeater.enabled = true;
        }
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

    public void Interrupt()
    {
        // Debug.LogWarning($"Interrupting at step {Academy.Instance.StepCount}");
        EpisodeInterrupted();
    }

    private void OnDestroy()
    {
        if (agentEvaluationManager)
        {
            agentEvaluationManager.TryInvokeOnEvalComplete();
        }
        
    }
}
