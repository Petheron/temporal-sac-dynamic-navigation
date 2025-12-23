using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;


public class AgentAction:MonoBehaviour, IModularAgentComponent
{
    [HideInInspector] public int inputSize = 1;
    [HideInInspector] public ComplexActionType complexActionType = ComplexActionType.Null;
    protected List<ModularActionType> actionTypes;
    protected Rigidbody agentRigidbody;
    protected EnvironmentParameters envParams;
    protected AgentProperties agentProps;
    protected ModularMovement movement;
    protected Transform cameraRotationTransform;
    protected List<Vector3> actionUsedPositions;

    public virtual void ProcessActionInput(float value) {}
    public virtual void ProcessActionInput(float value1, float value2) {} // Use for movement 

    public virtual void Perform() {}

    public virtual void Initialize(EnvironmentParameters environmentParameters, AgentProperties agentProperties)
    {
        envParams = environmentParameters;
        agentProps = agentProperties;
        agentRigidbody = GetComponent<Rigidbody>();
        movement = GetComponent<ModularMovement>();
        if (movement) cameraRotationTransform = movement.cameraFollowTransform;
        actionTypes = new List<ModularActionType>();

        InitializeAgentProperties();
        InitializeEnvironmentParameters();
        InitializeActionTypes();
        InitializeActionTimers();
        Reset();
    }

    protected virtual void InitializeActionTimers(){}
    protected virtual void InitializeActionTypes(){}

    public virtual void InitializeEnvironmentParameters(){}
    
    public virtual void InitializeAgentProperties(){}

    public virtual void Reset() {}

    public virtual List<float> CollectObservations()
    {
        return new List<float>();
    }

    public List<ModularActionType> GetActionTypes()
    {
        return actionTypes;
    }
    
    public List<Vector3> GetComplexActionPositions()
    {
        return complexActionType!=ComplexActionType.Null ? actionUsedPositions : null;
    }

    protected void StorePosition()
    {
        if (actionUsedPositions==null) return;
        actionUsedPositions.Add(transform.position);
    }

    protected float ClampInput(float value, float min = -1f, float max = 1f)
    {
        return Mathf.Clamp(value, min, max);
    }

    protected bool BinarizeBooleanInputValue(float value, float transitionValue = 0f)
    {
        return value > transitionValue;
    }

}
