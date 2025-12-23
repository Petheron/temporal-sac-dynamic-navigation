using System;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "AgentEvaluationManager", menuName = "AgentEvaluationManager", order = 0)]
public class AgentEvaluationManager : ScriptableObject
{
    [HideInInspector] public bool performEvaluation = false;
    [Tooltip("Do we evaluate all possible vertex combinations or assess edges of each vertex?")]
    public bool evaluateAll = false;
    public VertexSamplingManager vertexSamplingManager;

    private Dictionary<AgentType, Dictionary<int,Queue<Tuple<int,int>>>> remainingEvaluations;
    private List<AgentType> agentTypes;
    private int numAgentsPerType;
    
    public event Action OnEvaluationComplete;
    public event Action OnResultsSaved;
    public event Action OnTrajectoryReset;

    public void Initialize(List<AgentType> types, int numAgentsPerType)
    {
        agentTypes = types;
        this.numAgentsPerType = numAgentsPerType;
        remainingEvaluations = new Dictionary<AgentType, Dictionary<int, Queue<Tuple<int, int>>>>();
        
        if (vertexSamplingManager)
        {
            vertexSamplingManager.OnBeginEvaluationTrajectoryInitialization += InitializeTrajectories;
        }
    }

    private void InitializeTrajectories()
    {
        int numVertices = vertexSamplingManager.NumVertices();
        
        // Dictionary that maps vertex starting points to agent IDs
        // Each vertex can have one or more agent IDs.
        var vertToAgentDict = new Dictionary<int, List<int>>();
        for (int i = 0; i < numVertices; i++)
        {
            vertToAgentDict.Add(i, new List<int>());
            for (int j = 0; j < numAgentsPerType; j++)
            {
                // Accommodates for more agents than vertices AND more vertices than agents
                if (j % numVertices == i % numAgentsPerType)
                {
                    vertToAgentDict[i].Add(j);
                }
            }
        }
        
        foreach (var agentType in agentTypes)
        {
            var typeDict = new Dictionary<int, Queue<Tuple<int, int>>>();
            
            for (int i = 0; i < numAgentsPerType; i++)
            {
                typeDict.Add(i, new Queue<Tuple<int, int>>());
            }
            if (evaluateAll)
            {
                for (int i = 0; i < numVertices; i++)
                {
                    List<int> agentIDs = vertToAgentDict[i];
                    for (int idx = 0; idx < agentIDs.Count; idx++)
                    {
                        var queue = typeDict[agentIDs[idx]];
                        for (int j = idx; j < vertexSamplingManager.NumVertices(); j+=agentIDs.Count)
                        {
                            if(i==j) continue;
                            queue.Enqueue(new Tuple<int, int>(i,j));
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < numVertices; i++)
                {
                    List<int> agentIDs = vertToAgentDict[i];
                    for (int idx = 0; idx < agentIDs.Count; idx++)
                    {
                        var queue = typeDict[agentIDs[idx]];
                        var edges = vertexSamplingManager.GetVertex(i).GetEdges();
                        for (int e = idx; e < edges.Count; e+=agentIDs.Count)
                        {
                            var j = vertexSamplingManager.GetVertexIndex(edges[e]);
                            if(i==j) continue;
                            queue.Enqueue(new Tuple<int, int>(i,j));
                        }
                    }
                }
            }
            remainingEvaluations[agentType] = typeDict;
        }

        performEvaluation = true;
        OnTrajectoryReset?.Invoke();
    }

    private bool IsEvaluationComplete()
    {
        foreach (var pair in remainingEvaluations)
        {
            if (pair.Value.Count > 0) return false;
        }
        return true;
    }

    private void GetCurrentIndices(AgentType agentType, int agentIndex, out int currentStartIndex, out int currentGoalIndex)
    {
        if (!remainingEvaluations.TryGetValue(agentType, out var typeDict))
        {
            currentStartIndex = -1;
            currentGoalIndex = -1;
            return;
        }

        if (!typeDict.TryGetValue(agentIndex, out var remainingDeque))
        {
            currentStartIndex = -1;
            currentGoalIndex = -1;
            return;
        }
        
        if (!remainingDeque.TryDequeue(out var t))
        {
            currentStartIndex = -1;
            currentGoalIndex = -1;
            remainingEvaluations[agentType].Remove(agentIndex);
            return;
        }
        
        currentStartIndex = t.Item1;
        currentGoalIndex = t.Item2;
    }
    
    public bool SampleEvaluationNodes(AgentType agentType, int agentIndex, ref Vertex startVertex, ref Vertex goalVertex, ref int startIndex, ref int goalIndex)
    {
        if (!vertexSamplingManager)
        {
            return false;
        }
        
        GetCurrentIndices(agentType, agentIndex, out int currentStartIndex, out int currentGoalIndex);

        // If agentType has visited all vertices
        if (currentStartIndex >= vertexSamplingManager.NumVertices() || currentStartIndex<0)
        {
            startVertex = null;
            goalVertex = null;
            startIndex = -1;
            goalIndex = -1;
            return false;
        }
        
        startVertex = vertexSamplingManager.GetVertex(currentStartIndex);
        startIndex = currentStartIndex;
        
        goalVertex = vertexSamplingManager.GetVertex(currentGoalIndex);
        goalIndex = currentGoalIndex;

        if (!startVertex || !goalVertex)
        {
            Debug.Log($"Start {startIndex} is null?: {!startVertex}. Goal {goalIndex} is null?: {!goalVertex}");
        }
        
        return true;
    }

    public void TryInvokeOnEvalComplete()
    {
        if (IsEvaluationComplete())
        {
            OnEvaluationComplete?.Invoke();
        }
    }

    public void InvokeOnResultsSaved()
    {
        OnResultsSaved?.Invoke();
    }

}
