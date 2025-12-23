using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;
using Unity.MLAgents.Actuators;

public class JumpMovement:BaseMovement
{
    [Header("Jump Parameters")]
    public int numMidairJumps = 1;
    public bool preventMovementWhileJumping = false;
    public float jumpForce = 10.0f;
    public float jumpHeight = 2.2f;
    public float jumpDelay = 0.1f;
    
    protected int remainingMidairJumps;

    // Observables
    protected float jumpTimer;
    protected bool canJump => canGroundJump || CanAirJump;
    protected bool canGroundJump;
    protected bool CanAirJump => remainingMidairJumps>0;

    public override void ResetMovement(Vector3 position, Quaternion rotation)
    {
        base.ResetMovement(position, rotation);
        jumpTimer = 0f;
        canGroundJump = false;
        remainingMidairJumps = 0;

        complexActionPositions[ComplexActionType.Jump] = new List<Vector3>();
    }

    public override void SetDefaultMovementProperties(AgentProperties agentProperties)
    {
        base.SetDefaultMovementProperties(agentProperties);
        
        jumpForce = agentProperties.JUMP_FORCE;
        jumpHeight = agentProperties.JUMP_HEIGHT;
        jumpDelay = agentProperties.MINIMUM_JUMP_DURATION;
    }

    public override void ReceiveEnvironmentParameters(EnvironmentParameters parameters)
    {
        base.ReceiveEnvironmentParameters(parameters);
        
        numMidairJumps = (int) parameters.GetWithDefault("num_midair_jumps", numMidairJumps);
    }

    public override int SetNumObs()
    {
        int o = base.SetNumObs() + 1;
        numObs = o;
        return o;
    }

    protected override void DecrementTimers()
    {
        base.DecrementTimers();
        jumpTimer = jumpTimer > 0 ? jumpTimer - Time.fixedDeltaTime : 0f;
    }

    protected override void FixedUpdate()
    {
        base.FixedUpdate();

        if (!canApplyMoveInput)
        {
            if (IsGrounded && jumpTimer<=0f) canApplyMoveInput = true;
        }

        if (!IsGrounded && canGroundJump)
        {
            canGroundJump = false;
        }

        // reenable jump if touching ground
        if (IsGrounded && !canGroundJump && Velocity.y<0)
        {
            canGroundJump = true;
            jumpTimer = 0f;
            remainingMidairJumps = numMidairJumps;
        }
    }

    protected virtual void Jump(bool isJumpInput, float force, float multiplier=1.0f)
    {
        if (!isJumpInput) return;

        if (jumpTimer > 0f) return;

        if (!canJump) return;

        if (preventMovementWhileJumping)
        {
            // Apply input to jump direction if in mid-air
            if (!canApplyMoveInput)
            {
                canApplyMoveInput = true;
                base.Move();
            }

            canApplyMoveInput = false;
        }
        
        RemoveVerticalVelocity();
        if (useMomentum)
        {
            var jumpVector = multiplier*force * Vector3.up;
            agentRigidbody.AddForce(jumpVector,ForceMode.VelocityChange);
        }
        else
        {
            var initialVelocity = multiplier*Mathf.Sqrt(Mathf.Abs(2.0f * m_Gravity * jumpHeight));
            
            agentRigidbody.velocity += initialVelocity*Vector3.up;
        }
        
        jumpTimer = Mathf.Max(actionRepeatRate*Time.fixedDeltaTime, jumpDelay);

        complexActionPositions[ComplexActionType.Jump].Add(transform.position);

        if (canGroundJump)
        {
            canGroundJump = false;
        }
        else
        {
            remainingMidairJumps--;
        }
    }
    
    protected void RemoveVerticalVelocity()
    {
        var velocity = Velocity;
        velocity.y = 0f;
        agentRigidbody.velocity = velocity;
    }

    public override void ProcessActions(ActionSegment<float> act, float heuristicMouseScale = 1)
    {
        base.ProcessActions(act, heuristicMouseScale);

        var jumpInput = act[jumpIndex]>0f;
        Jump(jumpInput, jumpForce);
        
    }

    /// <summary>
    /// Sends movement observations to NavAgent.
    /// 9 observations from BaseMovement + 2 jump-relevant observations. [11 observations]
    /// </summary>
    /// <returns></returns>
    public override List<float> CollectVectorObservations()
    {
        var obs = base.CollectVectorObservations();
        obs.Add(canJump?1f:0f);
        // obs.Add(jumpTimer/jumpDelay);
        return obs;
    }
}
