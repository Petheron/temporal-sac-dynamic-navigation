using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;
using Unity.MLAgents;
using UnityEditor;

public enum VisibleTrajectory
{
    All,
    Success,
    Failure,
    Timeout,
    EucFail
}

public enum VisibleTrajectoryType
{
    All,
    DRL,
    NavMesh
}

public enum VisibleAgentType
{
    All,
    Base,
    Jump,
    FPS
}

public class TrajectoryVisualizer : MonoBehaviour
{
    [Header("LineRenderer Settings")]
    public TrajectoryComponent trajectoryPrefab;
    
    [ReadOnly] public float[] DRLTrajRates;
    private int[] numDRLTrajTypes;
    private Dictionary<int, float[]> results;
    private Dictionary<int, String> levelNames;
    private int resultIndex;
    private String resultsSubDir = "";
    public event Action OnTrajComplete;

    private GameObject trajectoryParentGO;
    private GameObject successGO;
    private GameObject failureGO;
    private GameObject timeoutGO;
    private GameObject eucFailGO;
    private GameObject navMeshGO;

    [Header("Visibility Options")] 
    public VisibleTrajectoryType visibleTrajectoryType = VisibleTrajectoryType.All;
    public VisibleTrajectory visibleTrajectories = VisibleTrajectory.All;
    public VisibleAgentType visibleAgentType = VisibleAgentType.All;
    public int startVertexIndex = -1;
    public int goalVertexIndex = -1;

    [SerializeField] private AgentEvaluationManager agentEvaluationManager;
    private LevelSpawner levelSpawner;


    private void Awake()
    {
        Initialize();

        if (agentEvaluationManager)
        {
            levelSpawner = FindObjectOfType<LevelSpawner>();
            agentEvaluationManager.OnEvaluationComplete += SaveResults;
            agentEvaluationManager.OnTrajectoryReset += Reset;

            results = new Dictionary<int, float[]>();
            levelNames = new Dictionary<int, String>();
            
            DateTime localDate = DateTime.Now;
            string date = $"{localDate.Year}{localDate.Month}{localDate.Day}";
            string time = $"{localDate.Hour}{localDate.Minute}{localDate.Second}";
            resultsSubDir = $"{date}_{time}";
        }
    }

    private void SaveResults()
    {
        print($"" +
              $"Success {DRLTrajRates[0]:F3}, " +
              $"Failure {DRLTrajRates[1]:F3}, " +
              $"Timeout {DRLTrajRates[2]:F3}, " +
              $"EucFail {DRLTrajRates[3]:F3}"
              );
        
        results.Add(resultIndex, DRLTrajRates);
        levelNames.Add(resultIndex, levelSpawner.LevelType.ToString());
        resultIndex++;

        // JsonSerializer serializer = new JsonSerializer();
        String path = Application.dataPath + "/Resources/Results/" + resultsSubDir + "/";

        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
            #if UNITY_EDITOR
                AssetDatabase.Refresh();
            #endif
            
        }
        
        String fileName = path + $"{resultsSubDir}_Results.csv";
        
        using StreamWriter writer = new StreamWriter(fileName);
        
        writer.WriteLine("Index,LevelType,Success,Failure,Timeout,EucFailure");

        foreach (var pair in results)
        {
            writer.WriteLine($"{pair.Key},{levelNames[pair.Key]},{pair.Value[0]},{pair.Value[1]},{pair.Value[2]},{pair.Value[3]}");
        }
        
        writer.Close();
        
        StartCoroutine(nameof(SaveScreenshot));
        
    }

    private IEnumerator SaveScreenshot()
    {
        if(!levelSpawner) yield break;
        string directory = "Assets/Resources/Results/" + resultsSubDir + "/";
        string levelName = levelSpawner.LevelType.ToString();
        DateTime localDate = DateTime.Now;
        string date = $"{localDate.Year}{localDate.Month}{localDate.Day}";
        string time = $"{localDate.Hour}{localDate.Minute}{localDate.Second}";
        ScreenCapture.CaptureScreenshot($"{directory}{date}_{time}_{levelName}.png");
        print($"Saved Screenshot to {directory}{date}_{time}_{levelName}.png");

        yield return new WaitForEndOfFrame();
        
        if (agentEvaluationManager)
        {
            agentEvaluationManager.InvokeOnResultsSaved();
        }
    }

    private void Initialize()
    {
        if(Academy.Instance.IsCommunicatorOn) Destroy(this);
        Reset();
    }

    private void Reset()
    {
        trajectoryParentGO = gameObject;
        numDRLTrajTypes = new int[4];
        DRLTrajRates = new float[4];
        
        if(successGO) Destroy(successGO);
        if(failureGO) Destroy(failureGO);
        if(timeoutGO) Destroy(timeoutGO);
        if(eucFailGO) Destroy(eucFailGO);
        if(navMeshGO) Destroy(navMeshGO);
    }
    
    public void StorePositions(int trajectoryID, List<Vector3> positions, EndCondition endCondition, Vector3 goalPos,
        Dictionary<ComplexActionType, List<Vector3>> complexActionPositions, AgentType agentType = AgentType.Base, TrajectoryType trajectoryType = TrajectoryType.DRL,
        int startIdx = 0, int goalIdx = 0)
    {
        if (trajectoryID < 0) return;
        if (positions == null) return;
        if (positions.Count == 0) return;
        
        // print($"Store Pos: Start Index: {startIdx}, Goal Index: {goalIdx}, cond: {endCondition.ToString()}");
        
        Trajectory trajectory =
            new Trajectory(trajectoryID, positions, endCondition, goalPos, complexActionPositions, agentType, trajectoryType, startIdx, goalIdx);

        if (trajectoryType == TrajectoryType.DRL)
        {
            numDRLTrajTypes[(int) endCondition] += 1;
            float total = numDRLTrajTypes.Sum();
            DRLTrajRates[(int)EndCondition.Success] = numDRLTrajTypes[(int) EndCondition.Success] / total;
            DRLTrajRates[(int)EndCondition.Failure] = numDRLTrajTypes[(int) EndCondition.Failure] / total;
            DRLTrajRates[(int)EndCondition.Timeout] = numDRLTrajTypes[(int) EndCondition.Timeout] / total;
            OnTrajComplete?.Invoke();
        }
        CreateLineRenderer(trajectory);
    }

    public void ToggleTrajectoryVisibility()
    {
        var disableList = GetGameObjectForVisibilitySetting(VisibleTrajectory.All, VisibleTrajectoryType.All);
        var enableList = GetGameObjectForVisibilitySetting(visibleTrajectories, visibleTrajectoryType);
        foreach (var go in disableList)
        {
            go.SetActive(false);
        }

        foreach (var go in enableList)
        {
            go.SetActive(true);

            for (int i = 0; i < go.transform.childCount; i++)
            {
                TrajectoryComponent trajectoryComponent = go.transform.GetChild(i)?.GetComponent<TrajectoryComponent>();
                ToggleTrajectoryVisibility(trajectoryComponent);
            }
        }
    }

    private List<GameObject> GetGameObjectForVisibilitySetting(VisibleTrajectory visibility, VisibleTrajectoryType visibleTrajeType)
    {
        List<GameObject> list = new List<GameObject>();

        if (visibleTrajeType is VisibleTrajectoryType.DRL or VisibleTrajectoryType.All)
        {
            switch (visibility)
            {
                case VisibleTrajectory.All:
                    if (successGO) list.Add(successGO);
                    if (failureGO) list.Add(failureGO);
                    if (timeoutGO) list.Add(timeoutGO);
                    if (eucFailGO) list.Add(eucFailGO);
                    break;
                case VisibleTrajectory.Success:
                    if (successGO) list.Add(successGO);
                    break;
                case VisibleTrajectory.Failure:
                    if (failureGO) list.Add(failureGO);
                    break;
                case VisibleTrajectory.Timeout:
                    if (timeoutGO) list.Add(timeoutGO);
                    break;
                case VisibleTrajectory.EucFail:
                    if (eucFailGO) list.Add(eucFailGO);
                    break;
            }
        }

        if (visibleTrajeType is VisibleTrajectoryType.NavMesh or VisibleTrajectoryType.All)
        {
            if (navMeshGO) list.Add(navMeshGO);
        }
        return list;
    }
    
    
    
    private GameObject HandleParentGameObject(Trajectory trajectory)
    {
        var cond = trajectory.endCondition;
        if (trajectory.trajectoryType == TrajectoryType.Navmesh)
        {
            if (!navMeshGO)
                navMeshGO = new GameObject
                {
                    name = $"{trajectory.trajectoryType.ToString()}Trajectories",
                    transform = {parent = trajectoryParentGO.transform}
                };
            return navMeshGO;
        }

        switch (cond)
        {
            case EndCondition.Success:
                if (!successGO)
                    successGO = new GameObject
                        {name = $"{cond.ToString()}Trajectories", transform = {parent = trajectoryParentGO.transform}};
                return successGO;
            case EndCondition.Failure:
                if (!failureGO)
                    failureGO = new GameObject
                        {name = $"{cond.ToString()}Trajectories", transform = {parent = trajectoryParentGO.transform}};
                return failureGO;
            case EndCondition.Timeout:
                if (!timeoutGO)
                    timeoutGO = new GameObject
                        {name = $"{cond.ToString()}Trajectories", transform = {parent = trajectoryParentGO.transform}};
                return timeoutGO;
            case EndCondition.EucFail:
                if (!eucFailGO)
                    eucFailGO = new GameObject
                        {name = $"{cond.ToString()}Trajectories", transform = {parent = trajectoryParentGO.transform}};
                return eucFailGO;
            default:
                return trajectoryParentGO;
        }
    }

    private void CreateLineRenderer(Trajectory trajectory)
    {
        var parent = HandleParentGameObject(trajectory);

        // GameObject trajectoryGO = Instantiate(trajectoryPrefab, parent.transform);
        // TrajectoryComponent trajectoryComponent = trajectoryGO.GetComponent<TrajectoryComponent>();
        TrajectoryComponent trajectoryComponent = Instantiate(trajectoryPrefab, parent.transform);
        trajectoryComponent.VisualizeTrajectory(trajectory);
        ToggleTrajectoryVisibility(trajectoryComponent);
    }

    private void ToggleTrajectoryVisibility(TrajectoryComponent trajectoryComponent)
    {
        if (!trajectoryComponent) return;
        trajectoryComponent.gameObject.SetActive(trajectoryComponent.IsVisibleByIndex(startVertexIndex, goalVertexIndex) &&
                                                 trajectoryComponent.IsVisibleByAgentType(visibleAgentType));
    }
}
