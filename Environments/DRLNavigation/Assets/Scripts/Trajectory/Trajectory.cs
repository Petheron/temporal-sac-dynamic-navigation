using System.Collections.Generic;
using UnityEngine;



public class Trajectory
{
    public int ID;
    public List<Vector3> positions;
    public EndCondition endCondition;
    public Vector3 goalPosition;
    public Dictionary<ComplexActionType, List<Vector3>> complexActionPositions;
    public AgentType agentType;
    public TrajectoryType trajectoryType;
    public int startIndex;
    public int goalIndex;

    public Trajectory(int id, List<Vector3> pos, EndCondition cond, Vector3 goalPos, Dictionary<ComplexActionType, List<Vector3>> complexPos, AgentType agentType, TrajectoryType trajType=TrajectoryType.DRL, int start = 0, int goal = 0)
    {
        ID = id;
        positions =   new List<Vector3>(pos);
        endCondition = cond;
        goalPosition = goalPos;
        complexActionPositions = complexPos;
        this.agentType = trajType == TrajectoryType.Navmesh?AgentType.Base: agentType;
        trajectoryType = trajType;
        startIndex = start;
        goalIndex = goal;
    }

    public override string ToString()
    {
        return $"ID: {ID}, Num Positions: {positions.Count}, EndCond: {endCondition.ToString()}, Agent Type: {agentType.ToString()}, TrajType: {trajectoryType.ToString()}, Start Idx: {startIndex}, Goal Idx: {goalIndex}";
    }
}