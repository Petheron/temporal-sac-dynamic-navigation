using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Actuators;

public class DashMovement:JumpMovement
{
    [Header("Dash Parameters")]
    public float dashSpeed = 60.0f;
    public float dashDuration = 0.25f;
    public float dashCooldown = 4f;
    [Space(10)]
    public float dashDurationTimer;
    public float dashCooldownTimer;
    public bool canDash = true;
    private Vector2 storedMoveInput;
    private Vector3 storedMoveDirection;


    public override void ResetMovement(Vector3 position, Quaternion rotation)
    {
        base.ResetMovement(position, rotation);
        canDash = true;
        dashDurationTimer = 0f;
        dashCooldownTimer = 0f;
    }

    protected override void DecrementTimers()
    {
        base.DecrementTimers();
        if (dashDurationTimer > 0)
        {
            dashDurationTimer -= Time.fixedDeltaTime;
            if (dashDurationTimer <= 0)
            {
                dashDurationTimer = 0f;
                EndDash();
            }
            else
            {
                Dash();
            }
        }

        if (dashCooldownTimer > 0)
        {
            dashCooldownTimer -= Time.fixedDeltaTime;
            if (dashCooldownTimer <= 0)
            {
                dashCooldownTimer = 0f;
                canDash = true;
            }
        }
    }

    protected void BeginDash(bool isDashInput)
    {
        if (!isDashInput) return;

        if (dashCooldownTimer > 0f) return;

        if (!canDash) return;

        dashDurationTimer = dashDuration;
        canDash = false;
        canApplyMoveInput = false;
        SetDrag(0f);
    }

    protected override void Move()
    {
        if (canApplyMoveInput)
        {
            base.Move();
        }
        else
        {
            storedMoveDirection = slopedForwardVector + slopedRightVector;
        }
    }
    
    protected override void ProcessMoveInputs(float strafeInput, float forwardInput)
    {
        var strafe = strafeInput;
        var forward = forwardInput;
        if (canApplyMoveInput)
        {
            storedMoveInput = new Vector2(strafeInput, forwardInput);
        }
        else
        {
            if (storedMoveInput.sqrMagnitude == 0)
            {
                storedMoveInput = Vector2.up;
            }
            else if (!storedMoveInput.sqrMagnitude.Equals(1.0f))
            {
                storedMoveInput = storedMoveInput.normalized;
            }
            strafe = storedMoveInput.x;
            forward = storedMoveInput.y;
        }
        
        // Accommodate for sloped surface.
        slopedRightVector = Vector3.Cross(-agentTransform.forward, GroundNormal)*strafe;
        slopedForwardVector = Vector3.Cross(agentTransform.right, GroundNormal)*forward;
    }

    private void Dash()
    {
        // Remove verticality/gravity effects
        // Remove drag if necessary
        // Apply acceleration

        RemoveVerticalVelocity();
        if(agentRigidbody.drag!=0f) SetDrag(0f);
        agentRigidbody.AddForce(dashSpeed*storedMoveDirection,ForceMode.Acceleration);
    }

    private void EndDash()
    {
        dashCooldownTimer = dashCooldown;
        canApplyMoveInput = true;
        SetDrag(IsGrounded?groundDrag:airDrag);
        
    }

    public override void ProcessActions(ActionSegment<float> act, float heuristicMouseScale = 1)
    {
        var dashInput = act.Length >= 6;
        if (dashInput)
        {
            dashInput = act[5]>0f;
            BeginDash(dashInput);
        }
        
        base.ProcessActions(act, heuristicMouseScale);

        if (dashDurationTimer > 0)
        {
            Dash();
        }
    }

    public override List<float> CollectVectorObservations()
    {
        var obs = base.CollectVectorObservations();
        obs.Add(canDash?1f:-1f);
        obs.Add(2*(dashDurationTimer/dashDuration)-1f);
        obs.Add(2*(dashCooldownTimer/dashCooldown)-1f);
        return obs;
    }
}
