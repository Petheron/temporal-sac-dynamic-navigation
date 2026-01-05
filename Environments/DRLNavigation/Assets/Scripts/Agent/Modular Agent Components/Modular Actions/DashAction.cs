using System;
using System.Collections.Generic;
using UnityEngine;

public class DashAction : AgentAction
{
    [Header("Dash Parameters")]
    private float dashDistance;
    private float dashDuration;
    private float dashCooldownDuration;
    private LayerMask dashLayerMask;
    
    private ActionTimer dashDurationTimer;
    private ActionTimer dashCooldownTimer;

    private bool isDashInputPressed;
    private bool isGroundBelowDashEndPosition;
    private Vector3 dashDirection;

    private bool CanDash => !IsDashing && !IsDashOnCooldown;
    private bool IsDashOnCooldown => !dashCooldownTimer.IsFinished;
    private bool IsDashing =>  !dashDurationTimer.IsFinished;
    
    private bool IsGrounded => movement.IsGrounded;
    private bool UseGroundCooldownReset => dashCooldownDuration <= 0f;
    private float Speed => dashDistance / dashDuration;

    public override void InitializeAgentProperties()
    {
        if(!ModularAgentProps) return;
        inputSize = 1;
        dashDistance = ModularAgentProps.DASH_DISTANCE;
        dashDuration = ModularAgentProps.DASH_DURATION;
        dashCooldownDuration = ModularAgentProps.DASH_COOLDOWN;
        dashLayerMask = ModularAgentProps.GROUNDED_MASK;
    }

    public override void InitializeEnvironmentParameters()
    {
        if (envParams == null) return;
        dashDistance = envParams.GetWithDefault("dash_distance", dashDistance);
        dashDuration = envParams.GetWithDefault("dash_duration", dashDuration);
        dashCooldownDuration = envParams.GetWithDefault("dash_cooldown", dashCooldownDuration);
    }

    protected override void InitializeActionTypes()
    {
        actionTypes.Add(ModularActionType.Dash);
        complexActionType = ComplexActionType.Dash;
    }

    protected override void InitializeActionTimers()
    {
        dashDurationTimer = new ActionTimer(dashDuration);
        dashCooldownTimer = new ActionTimer(UseGroundCooldownReset?1f:dashCooldownDuration);

        dashDurationTimer.OnFinished += EndDash; 
    }

    public override void ProcessActionInput(float value)
    {
        isDashInputPressed = BinarizeBooleanInputValue(value);
    }

    public override void Perform()
    {
        Dash();
    }

    private void Dash()
    {
        if (!isDashInputPressed) return;
        if (!CanDash) return;
        StartDash();
        
        StorePosition();
    }

    private void StartDash()
    {
        movement.checkForGround = false;
        movement.gravityEnabled = false;

        Vector3 dashVector = Speed * dashDirection;
        
        agentRigidbody.velocity.Set(0f,0f,0f);
        agentRigidbody.AddForce(dashVector, ForceMode.VelocityChange);
        
        dashDurationTimer.Start();
    }
    
    private void EndDash()
    {
        // movement.canTakeActions = true;
        movement.checkForGround = true;
        movement.gravityEnabled = true;
        agentRigidbody.velocity *= 0.5f;
        dashCooldownTimer.Start();
    }

    private void StoreExistGroundAtDashEnd()
    {
        // raycast toward dash end position
        // from either end of ray or first hit (retreat back along ray by 0.5m), cast ray down and store true if there is a hit.
        
        bool isHit = Physics.Raycast(transform.position, dashDirection, 
            out RaycastHit hit, dashDistance, dashLayerMask, QueryTriggerInteraction.Ignore);
        
        Vector3 endPosition = isHit ? 
                (hit.distance - 0.5f) * dashDirection:
                dashDistance*dashDirection;
        endPosition += transform.position;
        
        isGroundBelowDashEndPosition = Physics.Raycast(endPosition, Vector3.down, 100f, dashLayerMask, QueryTriggerInteraction.Ignore);

    }

    private void CalculateDashDirection()
    {
        Vector3 displacementVector = cameraRotationTransform.forward;
        dashDirection = displacementVector;
    }

    private void FixedUpdate()
    {
        HandleTimers(Time.fixedDeltaTime);
    }

    private void HandleTimers(float timePassed)
    {
        if (!movement) return;
        dashDurationTimer?.Decrement(timePassed);
        
        // Decrement timer normally if there is a cooldown
        if(!UseGroundCooldownReset) dashCooldownTimer?.Decrement(timePassed);
        else
        {
            // otherwise, check if grounded if a dash has been used.
            if (!IsDashing && IsDashOnCooldown && IsGrounded)
            {
                dashCooldownTimer.Decrement(100f);
            }
        }
    }

    public override List<float> CollectObservations()
    {
        CalculateDashDirection();
        StoreExistGroundAtDashEnd();
        
        List<float> obs = new List<float>
        {
            Convert.ToSingle(CanDash),
            Convert.ToSingle(IsDashing),
            Convert.ToSingle(isGroundBelowDashEndPosition)
        };
        if(!UseGroundCooldownReset) obs.Add(dashCooldownTimer.GetRemainingTime(true));
        return obs;
    }

    public override void Reset()
    {
        EndDash();
        dashDurationTimer.Reset();
        dashCooldownTimer.Reset();
        
        dashCooldownTimer.Start();
        
        isDashInputPressed = false;
        
        actionUsedPositions = new List<Vector3>();
    }

    // private void OnDrawGizmos()
    // {
    //     Gizmos.color = Color.white;
    //     Gizmos.DrawRay(transform.position, dashDistance*dashDirection);
    //     Gizmos.color = Color.red;
    //     Gizmos.DrawRay(transform.position + dashDistance*dashDirection, 10f*Vector3.down);
    // }
}
