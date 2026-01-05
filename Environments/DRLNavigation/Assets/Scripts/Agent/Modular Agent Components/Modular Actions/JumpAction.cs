using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.Serialization;

public class JumpAction : AgentAction
{
    [Header("Jump Parameters")] 
    [SerializeField] private float jumpVelocity;
    [SerializeField] private int numMidairJumps = 1;
    
    [Space(10f)]
    [SerializeField, ReadOnly] private int remainingJumps;
    [SerializeField, ReadOnly] private float minimumJumpDuration;
    [SerializeField, ReadOnly] private float coyoteTime = 0.2f;
    [SerializeField, ReadOnly] private bool isJumpInputPressed;
    
    private ActionTimer jumpTimer;
    private ActionTimer coyoteTimer;

    [SerializeField, ReadOnly] private float heightReached;
    private float initialHeight;

    public List<PassiveAgentAction> passiveAgentActions;

    private int MaxJumps => numMidairJumps + 1;
    private bool CanJump => remainingJumps > 0 && jumpTimer.IsFinished;
    private bool CanGroundJump => remainingJumps == MaxJumps;
    private bool CanCoyoteJump => !coyoteTimer.IsFinished;
    private bool IsGrounded => movement.IsGrounded;

    public override void Initialize(EnvironmentParameters environmentParameters, ModularAgentProperties modularAgentProperties)
    {
        base.Initialize(environmentParameters, modularAgentProperties);
        foreach (var passive in passiveAgentActions)
        {
            if(!passive) continue;
            passive.Initialize(environmentParameters,modularAgentProperties);
        }
        passiveAgentActions.RemoveAll(action => !action.usePassiveAction);
    }

    public override void InitializeAgentProperties()
    {
        if(!ModularAgentProps) return;
        inputSize = 1;
        jumpVelocity = ModularAgentProps.JUMP_VELOCITY;
        minimumJumpDuration = ModularAgentProps.MINIMUM_JUMP_DURATION;
        coyoteTime = ModularAgentProps.COYOTE_TIME_DURATION;
    }

    public override void InitializeEnvironmentParameters()
    {
        numMidairJumps = (int) envParams.GetWithDefault("num_midair_jumps", numMidairJumps);
    }
    
    protected override void InitializeActionTypes()
    {
        actionTypes.Add(ModularActionType.Jump);
        complexActionType = ComplexActionType.Jump;
    }

    protected override void InitializeActionTimers()
    {
        jumpTimer = new ActionTimer(Mathf.Max(movement.actionRepetition * Time.fixedDeltaTime, minimumJumpDuration));
        coyoteTimer = new ActionTimer(coyoteTime);
        coyoteTimer.OnFinished += ReduceRemainingJumps;
    }

    public override void ProcessActionInput(float value)
    {
        isJumpInputPressed = BinarizeBooleanInputValue(value); 
    }

    public override void Perform()
    {
        if (!isJumpInputPressed) return;
        foreach (var passiveAction in passiveAgentActions)
        {
            if (!passiveAction.IsActive()) continue;
            if(!jumpTimer.IsFinished) continue;
            remainingJumps++;
            passiveAction.Perform();
        }
        Jump();
        
    }

    private void Jump()
    {
        if (!CanJump) return;
        
        jumpTimer.Start();
        ReduceRemainingJumps();
        RemoveVerticalVelocity();
        ApplyJumpForce();
        
        StorePosition();
    }

    private void ApplyJumpForce()
    {
        initialHeight = movement.baseTransform.position.y;
        heightReached = -1f;
        agentRigidbody.AddForce(jumpVelocity * Vector3.up, ForceMode.VelocityChange);
    }

    private void HandleTimers(float timePassed)
    {
        jumpTimer?.Decrement(timePassed);
        coyoteTimer?.Decrement(timePassed);
    }

    private void FixedUpdate()
    {
        if (!movement) return;
        
        HandleTimers(Time.fixedDeltaTime);
        
        if (!IsGrounded && CanGroundJump && !CanCoyoteJump)
        {
            coyoteTimer.Start();
        }
        
        if (!IsGrounded && heightReached < 0f && agentRigidbody.velocity.y <= 0f)
        {
            heightReached = movement.baseTransform.position.y - initialHeight;
        }
        
        if (IsGrounded && (!CanGroundJump || CanCoyoteJump) && agentRigidbody.velocity.y <= 0f)
        {
            RefreshJumps();
        }
    }

    private void ReduceRemainingJumps()
    {
        remainingJumps--;
        if(CanCoyoteJump) coyoteTimer.Reset();
    }

    private void RemoveVerticalVelocity()
    {
        var velocity = agentRigidbody.velocity;
        velocity.y = 0f;
        agentRigidbody.velocity = velocity;
        
        agentRigidbody.AddForce(-agentRigidbody.GetAccumulatedForce().y*Vector3.up,ForceMode.Force);
    }

    public override List<float> CollectObservations()
    {
        List<float> obs = new List<float>
        {
            Convert.ToSingle(CanJump)
        };
        return obs;
    }

    private void RefreshJumps()
    {
        coyoteTimer.Reset();
        jumpTimer.Reset();
        remainingJumps = MaxJumps;
    }

    public override void Reset()
    {
        jumpTimer.Reset();
        remainingJumps = 0;
        isJumpInputPressed = false;
        actionUsedPositions = new List<Vector3>();
    }
}
