using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.Serialization;

public class MoveAction:AgentAction
{
    private enum MoveDirection
    {
        Forward = 0,
        Strafe = 1
    }

    private readonly MoveDirection[] moveDirections = {MoveDirection.Forward, MoveDirection.Strafe};
    
    [SerializeField] protected List<PassiveAgentAction> passiveActions; 
    
    public bool canMove = true;
    public bool useStrafe = true;

    public float acceleration;
    public float friction;
    public float airControl = 1.0f;
    public float turnAroundFactor = 1.0f;
    
    private float defaultAcceleration = 10f;
    private float defaultFriction;
    private float defaultAirControl;
    private float defaultTurnAroundFactor;
    
    private float[] inputs;
    private float[] currentSpeeds;
    private Vector3[] movementVectors;

    private bool IsGrounded => movement.IsGrounded;
    private float Acceleration => movement.currentMutator?.AccelerationModifier * acceleration ?? acceleration;
    private float Friction => movement.currentMutator?.FrictionModifier * friction ?? friction;
    private float AirControl => movement.currentMutator?.AirControlModifier * airControl ?? airControl;

    public float sprintModifier = 1.0f;


    public override void Initialize(EnvironmentParameters environmentParameters, AgentProperties agentProperties)
    {
        base.Initialize(environmentParameters, agentProperties);
        
        if (passiveActions!=null)
        {
            foreach (var passiveAction in passiveActions)
            {
                passiveAction.Initialize(envParams,agentProps);
            }
            passiveActions.RemoveAll(action => !action.usePassiveAction);
        }
    }

    public override void InitializeAgentProperties()
    {
        if (!agentProps) return;
        inputSize = useStrafe? 2:1;
        
        defaultAcceleration = agentProps.MOVE_ACCELERATION;
        defaultFriction = agentProps.MOVE_FRICTION;
        defaultAirControl = agentProps.MOVE_AIR_CONTROL;
        defaultTurnAroundFactor = agentProps.MOVE_TURNAROUND_FACTOR;
    }

    protected override void InitializeActionTypes()
    {
        actionTypes.Add(ModularActionType.Forward);
        if(useStrafe) actionTypes.Add(ModularActionType.Strafe);
    }

    public override void ProcessActionInput(float value)
    {
        ProcessActionInput(value,0f);
    }

    public override void ProcessActionInput(float value1, float value2)
    {
        if (!canMove) return;
        var forward = ClampInput(value1);
        var strafe = ClampInput(value2);

        float norm = Mathf.Max(Mathf.Sqrt(forward * forward + strafe * strafe),Mathf.Epsilon);
        
        inputs[(int)MoveDirection.Forward] =  Mathf.Abs(forward)>Mathf.Epsilon ? forward / norm : 0f;
        inputs[(int)MoveDirection.Strafe] =  Mathf.Abs(strafe)>Mathf.Epsilon ? strafe / norm : 0f;
    }
    
    public override void Perform()
    {
        if (!canMove) return;
        foreach (var passiveAction in passiveActions)
        {
            if (!passiveAction) continue;
            if (!passiveAction.IsActive()) continue;
            passiveAction.ProcessActionInput(inputs[0],inputs[1]);
            passiveAction.Perform();
            return;
        }
        CalculateMovementVectors();
        Move();
        ApplyFriction();
    }

    private void ApplyFriction()
    {
        if (!IsGrounded) return;
        Vector3 frictionVector = Vector3.zero;

        foreach (var direction in moveDirections)
        {
            frictionVector += CalculateFrictionComponent(direction);
        }

        agentRigidbody.AddForce(frictionVector, ForceMode.VelocityChange);
    }

    private Vector3 CalculateFrictionComponent(MoveDirection direction)
    {
        if (Mathf.Abs(inputs[(int) direction]) > 0f) return Vector3.zero;
        
        var movementVector = movementVectors[(int)direction];
        var currentSpeed = currentSpeeds[(int)direction];

        // Calculate friction
        float frictionAmount = Mathf.Min(Mathf.Abs(currentSpeed), Mathf.Abs(Friction));
        frictionAmount *= -Mathf.Sign(currentSpeed);

        return frictionAmount * movementVector;

    }

    private void CalculateMovementVectors()
    {
        // Calculate the movement direction (accommodates for sloped surfaces)

        foreach (var direction in moveDirections)
        {
            bool isStrafe = direction == MoveDirection.Strafe;
            Vector3 transformDirection = isStrafe ? -transform.forward : transform.right;
            Vector3 movementVector = Vector3.Cross(transformDirection, movement.groundNormal);
            float currentSpeed = Vector3.Dot(movementVector,  agentRigidbody.velocity);
            movementVectors[(int) direction] = movementVector;
            currentSpeeds[(int) direction] = currentSpeed;
        }
    }


    private void Move()
    {
        var moveVector = Vector3.zero;
        foreach (var direction in moveDirections)
        {
            moveVector += CalculateForceMovement(direction);
        }
        agentRigidbody.AddForce(moveVector, ForceMode.Acceleration);
    }

    private Vector3 CalculateForceMovement(MoveDirection direction)
    {
        Vector3 movementVector = movementVectors[(int) direction];
        float input = inputs[(int) direction];
        float currentSpeed = currentSpeeds[(int) direction];

        float magnitude = Acceleration * input;

        // no input pressed == do nothing
        if (Mathf.Abs(input) < Mathf.Epsilon) return Vector3.zero;

        // if changing direction == increased magnitude for fast turnaround, but only if grounded.
        if (Mathf.Abs(currentSpeed)>0f && Mathf.Sign(currentSpeed) * Mathf.Sign(input) < 0 && IsGrounded)
        {
            magnitude *= turnAroundFactor;
        }

        if (!IsGrounded) magnitude *= AirControl;

        if (direction == MoveDirection.Forward && input > 0)
        {
            magnitude *= sprintModifier;
        }

        return magnitude * movementVector;
    }

    public override List<float> CollectObservations()
    {
        var obs = new List<float>();
        foreach (var passiveAction in passiveActions)
        {
            if(passiveAction) obs.AddRange(passiveAction.CollectObservations());
        }
        return obs;
    }

    public override void Reset()
    {
        canMove = true;
        
        inputs = new []{0f,0f};
        currentSpeeds = new[] {0f, 0f};
        movementVectors = new[] {Vector3.zero, Vector3.zero};

        acceleration = defaultAcceleration;
        friction = defaultFriction;
        airControl = defaultAirControl;
        turnAroundFactor = defaultTurnAroundFactor;

        sprintModifier = 1.0f;
    }
}