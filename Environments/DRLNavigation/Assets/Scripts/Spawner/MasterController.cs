using System;
using LevelGeneration;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.Serialization;

public class MasterController : MonoBehaviour
{
    private AgentSpawner agentSpawner;
    private NavMeshAgentManager navMeshAgentManager;
    private LevelSpawner levelSpawner;
    private OccupancyMap occupancyMap;
    private TrajectoryPositionManager trajectoryPositionManager;

    [HideInInspector] public static bool isSpawning;

    public delegate void ControllerEvent();

    public event ControllerEvent SpawnLevel;
    public event ControllerEvent SpawnAgents;
    public event ControllerEvent InitializeOccupancyMap;
    public event ControllerEvent InitializeTrajectoryPositionsRecorder;
    
    [FormerlySerializedAs("trajectoryVisualizer")] public GameObject trajectoryVisualizerPrefab;

    private void Awake()
    {
        agentSpawner = GetComponent<AgentSpawner>();
        levelSpawner = GetComponent<LevelSpawner>();
        occupancyMap = GetComponent<OccupancyMap>();
        trajectoryPositionManager = GetComponent<TrajectoryPositionManager>();
        navMeshAgentManager = GetComponent<NavMeshAgentManager>();
    }

    private void Start()
    {
        InitializeTrajectoryPositionsRecorder += trajectoryPositionManager.InitializeRecorder;
        SpawnLevel += levelSpawner.SpawnLevel;
        SpawnAgents += agentSpawner.SpawnAgents;
        
        InitializeOccupancyMap += occupancyMap.InitializeOccupancyMap;

        if (!Academy.Instance.IsCommunicatorOn)
        {
            var trajectoryVisualizer = Instantiate(trajectoryVisualizerPrefab);
            trajectoryVisualizer.name = "TrajectoryVisualizer";
            navMeshAgentManager.trajectoryVisualizer = trajectoryVisualizer.GetComponent<TrajectoryVisualizer>();
            
            SpawnAgents += navMeshAgentManager.SpawnAgents;
        }
        else
        {
            DestroyImmediate(navMeshAgentManager);
        }
        InitializeTrajectoryPositionsRecorder?.Invoke();
        
        levelSpawner.Initialize();
        occupancyMap.Initialize();
        agentSpawner.Initialize();

        levelSpawner.LevelSpawned += BeginInitializeOccupancyMap;
        occupancyMap.OccupancyMapInitialized += BeginAgentSpawn;

        SpawnLevelIfReady();
    }

    private void BeginInitializeOccupancyMap()
    {
        InitializeOccupancyMap?.Invoke();
    }

    private void BeginAgentSpawn()
    {
        SpawnAgents?.Invoke();
    }

    private void SpawnLevelIfReady()
    {
        if (CheckEnvironmentReady())
        {
            isSpawning = true;
            SpawnLevel?.Invoke();
        }
    }
    
    private bool CheckEnvironmentReady()
    {
        return agentSpawner.IsReady && levelSpawner.IsReady && occupancyMap.IsReady && !isSpawning;
    }
}