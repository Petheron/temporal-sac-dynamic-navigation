using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Random = UnityEngine.Random;

public enum LevelType
{
    Navigation,
    Winding,
    Tower,
    JumpPad,
    CannonLevel,
    MovingPlatV2,
    MovingPlatV3,
    MovingPlatV2Indep,
    MovingPlatV3Indep,
    Crumbling,
    WallRide,
    WallClimb,
    WallJump,
    SlowLevel,
    WaterLevel,
    IceLevel,
    PCG
}

public class LevelSpawner : MonoBehaviour
{
    [Header("Level Spawner Properties")]
    [SerializeField] private Level[] levelArray;
    [SerializeField] private LevelType levelType = LevelType.Navigation;
    
    [Header(" Level Sequence Properties")]
    [SerializeField] private bool useLevelSequence;
    [SerializeField] private List<LevelType> levelSequence;
    private int sequenceDelay = 1000;
    private int previousStepCount;
    
    // Private variables
    private int sequenceIndex;
    private Level currentLevel;
    private CameraController cameraController;
    private OccupancyMap occupancyMap;
    private AgentSpawner agentSpawner;

    [Header("External References")]
    private EnvironmentParameters envParams;
    [SerializeField] private AgentProperties agentProperties;
    [SerializeField] private VertexSamplingManager vertexSamplingManager;
    [SerializeField] private AgentEvaluationManager agentEvaluationManager;
    
    // Events
    public MasterController.ControllerEvent LevelSpawned;
    public bool IsReady { get; private set; }

    private bool hasHadFirstReset;
    
    private bool IsLoadNextLevel => sequenceDelay>0 
                                    && Academy.Instance.StepCount > 0
                                    && Academy.Instance.StepCount % sequenceDelay == 0
                                    && Academy.Instance.StepCount != previousStepCount;

    public LevelType LevelType => levelType;

    public void Initialize()
    {
        SetSpawnerProperties();
        SetSequenceProperties();

        if (vertexSamplingManager)
        {
            vertexSamplingManager.OnGraphMade += InvokeLevelSpawned;
        }

        if (agentEvaluationManager)
        {
            agentEvaluationManager.OnResultsSaved += LoadNextLevel;
        }
        
        Academy.Instance.OnEnvironmentReset += ResetLevelSpawner;
    }
    
    private void Awake()
    {
        cameraController = GetComponent<CameraController>();
    }

    private void SetSpawnerProperties()
    {
        envParams = Academy.Instance.EnvironmentParameters;
        IsReady = true;
        occupancyMap = GetComponent<OccupancyMap>();
        agentSpawner = GetComponent<AgentSpawner>();
    }

    private void SetSequenceProperties()
    {
        useLevelSequence = envParams.GetWithDefault("use_level_sequence", useLevelSequence?1:0)>0;
        sequenceDelay = (int)envParams.GetWithDefault("level_sequence_delay", agentProperties.LEVEL_SEQUENCE_DELAY);

        if (levelSequence.Count == 0) useLevelSequence = false;

        if (useLevelSequence) levelType = levelSequence[0];
    }

    private void FixedUpdate()
    {
        if(IsLoadNextLevel) LoadNextLevel();
    }

    public void SpawnLevel()
    {
        if(!useLevelSequence) levelType = (LevelType)envParams.GetWithDefault("level_type", (int) levelType);
        SpawnLevel(levelType);
    }

    /// <summary>
    /// Spawn a new level according to the given LevelType.
    /// </summary>
    /// <param name="newLevelType"></param>
    private void SpawnLevel(LevelType newLevelType)
    {
        // Pause time to prevent physics and Academy from progressing whilst level is spawning and agent is resetting.
        TimeScaleControl.TogglePauseState(true);
        
        // Interrupt before clearing level to prevent final observation that occurs post interrupt from involving a null vertices and incorrect observations.
        agentSpawner.InterruptAgents();
        
        ClearCurrentLevel();
        Level levelPrefab = levelArray[(int) newLevelType];
        currentLevel = Instantiate(levelPrefab, Vector3.zero, Quaternion.identity);
        currentLevel.name = levelPrefab.name;
        
        // Get the Level class (all levels should have this), and initialize all components and cameras.
        currentLevel.Initialize(envParams, agentProperties, occupancyMap, cameraController);
    }

    /// <summary>
    /// When level is finished spawning (when Digraph is made), invoke the LevelSpawned event.
    /// </summary>
    private void InvokeLevelSpawned()
    {
        LevelSpawned?.Invoke();
    }

    /// <summary>
    /// Destroy the current level and remove all cameras
    /// </summary>
    private void ClearCurrentLevel()
    {
        if (!currentLevel) return;
        cameraController.ClearCameras();
        DestroyImmediate(currentLevel.gameObject);
        currentLevel = null;
    }
    
    /// <summary>
    /// Load the next level in the sequence of levels.
    /// </summary>
    private void LoadNextLevel()
    {
        if (!useLevelSequence) return;
        if (levelSequence.Count == 0) return;

        previousStepCount = Academy.Instance.StepCount;
        
        sequenceIndex = (sequenceIndex + 1) % levelSequence.Count;

        if (sequenceIndex == levelSequence.Count - 1 && agentEvaluationManager)
        {
            agentEvaluationManager.OnResultsSaved -= LoadNextLevel;
        }

        if (agentEvaluationManager)
        {
            SetSeed();
        }
        
        levelType = levelSequence[sequenceIndex];
        
        Debug.Log($"Loading {levelType} environment at env step: {Academy.Instance.StepCount}");
        
        SpawnLevel(levelType);
    }
    
    /// <summary>
    /// Set the seed state of the random generator once per environment reset.
    /// This is shared by all agents and used when randomly selecting start/goal vertex.
    /// </summary>
    private void SetSeed()
    {
        int seed = (int)envParams.GetWithDefault("seed", agentProperties.DEFAULT_SEED);
        Random.InitState(seed);
        Academy.Instance.InferenceSeed = seed;
    }

    /// <summary>
    /// Reset level when environment resets but only for resets after the first time to allow for MasterController to initialize everything.
    /// </summary>
    private void ResetLevelSpawner()
    {
        // Debug.Log($"\nReset Level at Step Count: {Academy.Instance.StepCount}, Total Step Count: {Academy.Instance.TotalStepCount}\n");
        SetSeed();
        if (!hasHadFirstReset)
        {
            hasHadFirstReset = true;
            return;
        }

        if (useLevelSequence)
        {
            sequenceIndex = 0;
            levelType = levelSequence[sequenceIndex];
        }
        
        SpawnLevel(levelType);
    }
}
