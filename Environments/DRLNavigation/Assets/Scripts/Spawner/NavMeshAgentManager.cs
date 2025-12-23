using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class NavMeshAgentManager:MonoBehaviour
{
    public int numAgents = 0;
    public GameObject navmeshAgentPrefab;

    private List<GameObject> agentList;
    private GameObject agentContainer;

    public VertexSamplingManager vertexSamplingManager;
    public AgentProperties agentProperties;
    [HideInInspector] public TrajectoryVisualizer trajectoryVisualizer;
    
    public AgentEvaluationManager agentEvaluationManager;
    private bool evaluateAll; 
    
    private int startIndex;
    private int goalIndex;
    private int edgeIndex = 0;
    private int numVertices;
    private int sampleFreq = 1;

    private void Awake()
    {
        SetProperties();
    }

    private void SetProperties()
    {
        if (Academy.Instance.IsCommunicatorOn)
        {
            numAgents = 0;
            return;
        }
        agentContainer = new GameObject
        {
            name = "NavAgentContainer"
        };
        sampleFreq = agentProperties.ACTION_REPETITION;
        
        if (agentEvaluationManager)
        {
            if(!agentEvaluationManager.performEvaluation) Destroy(this);
            evaluateAll = agentEvaluationManager.evaluateAll;
            
        }
    }
    
    private void ClearExistingAgents()
    {
        if (agentList == null) return;
        if (agentList.Count <= 0) return;
        foreach (var agent in agentList)
        {
            if(agent) Destroy(agent);
        }
    }

    public void SpawnAgents()
    {
        if (!agentContainer) return;
        if (!agentEvaluationManager) return;
        if (!agentEvaluationManager.performEvaluation) return;
        ClearExistingAgents();
        agentList = new List<GameObject>();

        for (int i = 0; i < numAgents; i++)
        {
            GameObject agent = Instantiate(navmeshAgentPrefab, Vector3.zero, Quaternion.identity, agentContainer.transform);
            agent.name = $"NavmeshAgent_{i}";
            NavigationMeshAgent navigationMeshAgent = agent.GetComponent<NavigationMeshAgent>();
            if (navigationMeshAgent)
            {
                navigationMeshAgent.manager = this;
                navigationMeshAgent.sampleFreq = sampleFreq;
                navigationMeshAgent.agentID = i;
                navigationMeshAgent.numAgents = numAgents;
                navigationMeshAgent.trajectoryVisualizer = trajectoryVisualizer;
            }
            agentList.Add(agent);
        }
        
        numVertices = vertexSamplingManager.NumVertices();
    }

    public Tuple<int, Vector3, int, Vector3> GetStartGoalPositions()
    {
        
        if (!vertexSamplingManager) return null;

        if (evaluateAll && goalIndex == startIndex)
        {
            VertexSamplingManager.IncrementVertexIndices(ref startIndex, ref goalIndex, numVertices);
        }
        
        if (goalIndex >= numVertices) return null;
        if (startIndex >= numVertices) return null;

        int oldStartIdx = startIndex;
        int oldGoalIdx = goalIndex;
        Vertex startVertex = vertexSamplingManager.GetVertex(startIndex);
        Vector3 startPosition = startVertex.GetPosition();
        Vertex goalVertex;

        if (evaluateAll)
        {
            goalVertex = vertexSamplingManager.GetVertex(goalIndex);
            VertexSamplingManager.IncrementVertexIndices(ref startIndex, ref goalIndex, numVertices);
        }
        else
        {
            goalVertex = startVertex.GetEdgeVertex(edgeIndex);
            oldGoalIdx = vertexSamplingManager.GetVertexIndex(goalVertex);
            VertexSamplingManager.IncrementEdgeIndex(ref startIndex, ref edgeIndex, startVertex.NumEdges() );
        }
        Vector3 goalPosition = goalVertex.GetPosition();
        

        return new Tuple<int, Vector3, int, Vector3>(oldStartIdx, startPosition, oldGoalIdx, goalPosition);
    }
}
