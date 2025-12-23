using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.Barracuda;
using Unity.MLAgents.Policies;

public enum SpawnerMode
{
    Training,
    Evaluating
}

public class AgentSpawner : MonoBehaviour
{
    [Header("General Properties")] 
    [SerializeField] private SpawnerMode spawnerMode = SpawnerMode.Training;
    [SerializeField] private List<AgentType> agentTypesToSpawn = new() {};
    [SerializeField] private int numAgentsPerSpawnType = 1;
    
    [Space(10)]
    
    [SerializeField] private GameObject[] agentPrefabArray;
    [SerializeField] private NNModel[] agentModelArray;
    
    private float agentSize = 0f;
    private float agentSpacing = 0f;
    [SerializeField, ReadOnly] private List<GameObject> agentList;
    private EnvironmentParameters envParams;
    
    [SerializeField] private AgentEvaluationManager agentEvaluationManager;
    
    // Camera parameters
    private CameraController cameraController;

    // Initialization Parameters
    private int numAgentsInitialized;
    private bool hasDisposed;
    
    // Auto-properties
    public bool IsReady { get; private set; }
    public bool IsEvaluating => spawnerMode == SpawnerMode.Evaluating;

    // Events
    public event Action OnAcademySteppingEnabled;


    public void Initialize()
    {
        Academy.Instance.AutomaticSteppingEnabled = false;
        SetProperties();
        agentList = new List<GameObject>();
        
        if (!IsEvaluating) return;
        
        if (Academy.Instance.IsCommunicatorOn || !agentEvaluationManager)
        {
            spawnerMode = SpawnerMode.Training;
            return;
        }

        if (agentEvaluationManager)
        {
            agentEvaluationManager.Initialize(agentTypesToSpawn, numAgentsPerSpawnType);
        }
        
    }

    private void SetProperties()
    {
        envParams = Academy.Instance.EnvironmentParameters;
        IsReady = true;
        cameraController = GetComponent<CameraController>();

        if (envParams != null)
        {
            agentTypesToSpawn = new List<AgentType>()
                {(AgentType) envParams.GetWithDefault("agent_type", (int) agentTypesToSpawn[0])};
            numAgentsPerSpawnType = (int) envParams.GetWithDefault("num_agents",numAgentsPerSpawnType);
        }
        
    }
    
    public void SpawnAgents()
    {
        SpawnAgents(numAgentsPerSpawnType);
        MasterController.isSpawning = false;
    }

    private void SpawnAgents(int nAgents)
    {
        // Debug.Log("Spawning Agents");
        // Reset existing agents and return
        if (agentList.Count>0)
        {
            ResetAgents();
            TimeScaleControl.TogglePauseState(false);
            return;
        }
        
        // print("is Spawning agents");
        
        int totalNumAgents = nAgents * agentTypesToSpawn.Count;
        
        // agentList = new GameObject[totalNumAgents];
        int mod = Mathf.CeilToInt(Mathf.Sqrt(totalNumAgents));
        
        float agentWidth = (agentSize+agentSpacing);
        float halfWidth = mod*agentWidth;

        if (!agentTypesToSpawn.Contains(AgentType.Modular))
        {
            Academy.Instance.AutomaticSteppingEnabled = true;
        }

        foreach (var agentType in agentTypesToSpawn)
        {
            var agentPrefab = agentPrefabArray[(int) agentType];
            var model = agentModelArray[(int) agentType];
            
            for(int i = 0;i<nAgents;i++)
            {
                GameObject agentGameObject = Instantiate(agentPrefab,Vector3.zero,Quaternion.identity,transform.parent);
                agentGameObject.name = agentPrefab.name + i;
                float xPos = 2.0f*(i%mod)*agentWidth -halfWidth + agentWidth;
                float yPos = halfWidth - 2.0f*((int)(i/mod))*agentWidth - agentWidth;
                agentGameObject.transform.localPosition = new Vector3(xPos,yPos,0f);
                // agentList[i] = agentGameObject;
                agentList.Add(agentGameObject);
            
                var navAgent = agentGameObject.GetComponentInChildren<MonolithicNavAgent>();
                if (navAgent)
                {
                    navAgent.agentID = i;
                    navAgent.numAgents = nAgents;
                    navAgent.agentType = agentType;
                    navAgent.agentEvaluationManager = spawnerMode==SpawnerMode.Evaluating ? agentEvaluationManager: null;
                }
                else
                {
                    var modAgent = agentGameObject.GetComponentInChildren<ModularNavAgent>();
                    if (modAgent)
                    {
                        modAgent.agentID = i;
                        modAgent.numAgents = nAgents;
                        modAgent.agentType = agentType;
                        modAgent.agentSpawner = this;
                        modAgent.agentEvaluationManager = spawnerMode==SpawnerMode.Evaluating ? agentEvaluationManager: null;
                        modAgent.OnAgentPostInitialize += IncrementInitializedAgentsCount;
                        OnAcademySteppingEnabled += modAgent.AllowEnable;
                        modAgent.InitializeAgentComponents();
                        modAgent.SetCameraController(cameraController);
                    }
                }

                if (spawnerMode == SpawnerMode.Evaluating)
                {
                    var behaviorParams = agentGameObject.GetComponentInChildren<BehaviorParameters>();
                    behaviorParams.Model = model;
                    behaviorParams.InferenceDevice = InferenceDevice.GPU;
                    behaviorParams.BehaviorType = BehaviorType.InferenceOnly;
                }
            }
            // if(mainCam) mainCam.orthographicSize = halfWidth;
        }
        TimeScaleControl.TogglePauseState(false);
    }

    // private void ClearAgents(){
    //     if(agentList==null) return;
    //
    //     for(int i =0;i<agentList.Count;i++){
    //         GameObject g = agentList[i];
    //         cameraController.RemoveCamera(g.GetComponentInChildren<Camera>());
    //         agentList[i] = null;
    //         if(g) Destroy(g);
    //     }
    //     agentList.Clear();
    // }

    public void RemoveAgent(GameObject agentGO)
    {
        agentList.Remove(agentGO);
    }

    /// <summary>
    /// Resets all agents that exist. Used when a new level is spawned.
    /// Forces all agents to be interrupted.
    /// </summary>
    public void InterruptAgents()
    {
        if (agentList == null) return;
        if (agentList.Count == 0) return;
        if (agentList.Count == 0) return;
        
        foreach (var go in agentList)
        {
            // if (!go) continue;
            var navAgent = go.GetComponentInChildren<MonolithicNavAgent>();
            if (navAgent)
            {
                navAgent.Interrupt();
                continue;
            }
            var modAgent = go.GetComponentInChildren<ModularNavAgent>();
            if (modAgent)
            {
                modAgent.SetCameraController(cameraController);
                modAgent.Interrupt();
            }
        }
        // Debug.Log("Agents interrupted");
    }

    public void ResetAgents()
    {
        if (agentList == null) return;
        if (agentList.Count == 0) return;
        
        // Debug.Log("Resetting agents");
        
        foreach (var go in agentList)
        {
            var navAgent = go.GetComponentInChildren<MonolithicNavAgent>();
            if (navAgent)
            {
                navAgent.Reset();
                continue;
            }
            var modAgent = go.GetComponentInChildren<ModularNavAgent>();
            if (modAgent)
            {
                modAgent.SetCameraController(cameraController);
                modAgent.Reset();
            }
        }
    }

    public void IncrementInitializedAgentsCount()
    {
        numAgentsInitialized++;
        if (numAgentsInitialized < numAgentsPerSpawnType) return;
        numAgentsInitialized = 0;
        Academy.Instance.AutomaticSteppingEnabled = true;
        OnAcademySteppingEnabled?.Invoke();
    }
}
