using System;
using System.Collections.Generic;
using UnityEngine;

public class SprintAction : AgentAction
{
    [Header("Sprint Parameters")] 
    [SerializeField] private bool isSprintToggle;
    [SerializeField] private float sprintModifier;
    [SerializeField] private bool isSprinting;
    [SerializeField] private float sprintDurationMinimum = 0.2f;
    private ActionTimer sprintTimer;

    [SerializeField] private MoveAction moveAction;

    private bool isSprintInputPressed;

    private bool CanSprint => sprintTimer.IsFinished;

    public override void InitializeAgentProperties()
    {
        if(!agentProps) return;
        inputSize = 1;
        isSprintToggle = agentProps.SPRINT_IS_TOGGLE;
        sprintModifier = agentProps.SPRINT_MODIFIER;
        sprintDurationMinimum = agentProps.SPRINT_DURATION_MINIMUM;
    }
    
    protected override void InitializeActionTypes()
    {
        actionTypes.Add(ModularActionType.Sprint);
        complexActionType = ComplexActionType.Sprint;
    }

    protected override void InitializeActionTimers()
    {
        sprintTimer = new ActionTimer(sprintDurationMinimum);
    }

    public override void ProcessActionInput(float value)
    {
        isSprintInputPressed = BinarizeBooleanInputValue(value);
    }

    public override void Perform()
    {
        Sprint();
    }

    private void Sprint()
    {
        // if (!movement.IsGrounded) return;
        if (!CanSprint) return;

        if (isSprintToggle)
        {
            if (!isSprintInputPressed) return;
            if (isSprinting) EndSprint();
            else StartSprint();
            return;
        }

        if (isSprintInputPressed && !isSprinting) StartSprint();
        else if (!isSprintInputPressed && isSprinting) EndSprint();
        
    }

    private void StartSprint()
    {
        isSprinting = true;
        sprintTimer.Start();
        if (moveAction) moveAction.sprintModifier = sprintModifier;
        StorePosition();
    }
    
    private void EndSprint()
    {
        isSprinting = false;
        sprintTimer.Start();
        moveAction.sprintModifier = 1f;
        StorePosition();
    }

    private void FixedUpdate()
    {
        sprintTimer?.Decrement(Time.fixedDeltaTime);
    }

    public override List<float> CollectObservations()
    {
        List<float> obs = new List<float>
        {
            Convert.ToSingle(CanSprint),
            Convert.ToSingle(isSprinting)
        };
        return obs;
    }

    public override void Reset()
    {
        sprintTimer.Reset();
        isSprintInputPressed = false;
        actionUsedPositions = new List<Vector3>();
    }
}
