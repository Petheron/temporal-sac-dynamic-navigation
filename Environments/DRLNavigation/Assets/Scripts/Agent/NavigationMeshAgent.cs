using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.AI;

public class NavigationMeshAgent : MonoBehaviour
{
    public Transform goalTransform;
    [HideInInspector] public NavMeshAgentManager manager;

    private NavMeshAgent navMeshAgent;
    private int startIdx;
    private int goalIdx;
    private Vector3 startPosition;
    private Vector3 goalPosition;
    private float maxGoalDistance;
    private List<Vector3> trajPosList;
    public TrajectoryVisualizer trajectoryVisualizer;
    private int stepCount;
    [HideInInspector] public int sampleFreq = 1;

    [HideInInspector] public int agentID;
    [HideInInspector] public int numAgents;
    private int trajCount; 

    private void Awake()
    {
        navMeshAgent = GetComponentInChildren<NavMeshAgent>();
    }

    private void Start()
    {
        Initialize();
    }

    private void Reset()
    {
        trajPosList = new List<Vector3>();
        navMeshAgent.velocity = Vector3.zero;
        stepCount = 0;
        if (GetPositions())
        {
            Vector3 offset = (navMeshAgent.height / 2f) * Vector3.up;
            navMeshAgent.Warp(startPosition+offset);
            goalTransform.position = goalPosition;
            navMeshAgent.SetDestination(goalPosition);
        }
        else
        {
            Destroy(gameObject);
        }
        
    }

    private void Initialize()
    {
        Reset();
    }

    private bool GetPositions()
    {
        if (!manager) return false;
        var positions = manager.GetStartGoalPositions();
        if (positions==null)
        {
            return false;
        }

        startIdx = positions.Item1;
        startPosition = positions.Item2;
        goalIdx = positions.Item3;
        goalPosition = positions.Item4;
        
        // Debug.Log($"NavMeshAgent Trajectory: Start: {startIdx}, Goal: {goalIdx}");
        
        return true;
    }

    private void FixedUpdate()
    {
        Vector3 currentPos = navMeshAgent.transform.position;
        stepCount++;
        if (stepCount % sampleFreq==0)
        {
            trajPosList.Add(currentPos);
        }
        
        if (Vector3.SqrMagnitude( navMeshAgent.transform.position - goalPosition) <= 1.0f)
        {
            trajPosList.Add(currentPos);
            int id = agentID + numAgents * trajCount;
            if(trajectoryVisualizer) trajectoryVisualizer.StorePositions(id, trajPosList, EndCondition.Success, goalPosition,null,  AgentType.Base, TrajectoryType.Navmesh, startIdx, goalIdx);
            trajCount++;
            Reset();
        }
    }
}
