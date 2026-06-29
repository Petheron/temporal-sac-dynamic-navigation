using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Serialization;

public class ModularMovement : MonoBehaviour, IModularAgentComponent
{
    public Transform cameraFollowTransform;
    public PlayerActionsManager playerActionsManager;
    public Grounding grounding;
    
    [Header("Action Parameters")]
    public int actionBitMask;
    public List<AgentAction> actions;
    [ReadOnly] public List<ModularActionType> actionTypes;

    [Header("Movement Behaviour Toggles")] 
    public bool observeVel = true;
    public bool observeAccel = true;
    public bool observeMutator = true;
    private ObservedRotationType observedRotationType = ObservedRotationType.YawPitch;
    private VectorNormalizationType vectorNormalizationType = VectorNormalizationType.Componentwise;
    private VectorSpatialOrientationType vectorSpatialOrientationType = VectorSpatialOrientationType.WorldSpace;
    [HideInInspector] public bool rotateTransform = true;
    [HideInInspector] public bool canTakeActions = true;
    [HideInInspector] public bool gravityEnabled = true;
    [HideInInspector] public bool checkForGround = true;

    [ReadOnly] public MutatorType currentMutatorType;
    [ReadOnly] public MutatorType newMutatorType;
    private bool hasStoredMutator;

    [Header("Grounded Parameters")] 
    public Transform baseTransform;
    [ReadOnly] public Vector3 groundNormal;
    [SerializeField, ReadOnly] private bool isGrounded = false;
    private float groundCheckOffset = 0.41f;
    private float groundCheckRadius = 0.44f;
    private LayerMask groundCheckMask;
    
    private Rigidbody agentRigidbody;
    [SerializeField]private Quaternion rotationQuaternion = Quaternion.identity;
    private float defaultGravity;
    [HideInInspector] public float gravityModifier = 1f;
    public float fallingMultiplier = 1.5f;
    [SerializeField] private float drag;
    [ReadOnly] public float dragModifier = 1.0f;

    [HideInInspector] public int actionRepetition=1;
    private Vector3 oldVelocity;
    [HideInInspector] public float velocityLimit;
    public bool IsGrounded => isGrounded;
    private float AccelLimit => 2 * velocityLimit / actionRepetition*Time.fixedDeltaTime;
    public Vector3 Velocity =>  agentRigidbody? agentRigidbody.velocity: Vector3.zero;
    private Vector3 Acceleration => (Velocity - oldVelocity) / actionRepetition*Time.fixedDeltaTime;
    public Vector3 RelativeVelocity => WorldToLocal.MultiplyVector(Velocity);
    private Vector3 RelativeAcceleration => WorldToLocal.MultiplyVector(Acceleration);
    private float Gravity => gravityModifier*defaultGravity;
    private Vector3 ForwardVector => cameraFollowTransform.forward;
    public Quaternion Rotation => rotationQuaternion;
    private float Yaw => Mathf.Repeat(Rotation.eulerAngles.y, 360f) / 180f - 1;
    private float Pitch
    {
        get
        {
            float pitch = Rotation.eulerAngles.x;
            pitch = pitch > 180f ? pitch - 360f : pitch;
            return -pitch / 90f;
        }
    }
    
    public Matrix4x4 WorldToLocal
    {
        get
        {
            if(Rotation.Equals(new Quaternion(0,0,0,0))) return Matrix4x4.TRS(transform.position, Quaternion.identity, Vector3.one).inverse;
            return Matrix4x4.TRS(transform.position, Rotation, Vector3.one).inverse;
        }
    }

    private float AirControl => modularAgentProps.MOVE_AIR_CONTROL;
    private float Drag => drag * dragModifier;


    private EnvironmentParameters envParams;
    private ModularAgentProperties modularAgentProps;

    public event Action OnInitialized;

    private void Awake()
    {
        agentRigidbody = GetComponent<Rigidbody>();
        if (agentRigidbody)
        {
            agentRigidbody.useGravity = false;
            agentRigidbody.freezeRotation = true;
            agentRigidbody.drag = 0f;
            agentRigidbody.angularDrag = 0f;
        }
    }

    public void Initialize(EnvironmentParameters environmentParameters, ModularAgentProperties modularAgentProperties)
    {
        envParams = environmentParameters;
        modularAgentProps = modularAgentProperties;
        InitializeAgentProperties();
        InitializeEnvironmentParameters();
        InitializeAgentActions();
        OnInitialized?.Invoke();
    }

    private void InitializeAgentActions()
    {
        MaskList(actions, actionBitMask);
        
        actionTypes = new List<ModularActionType>();

        foreach (var agentAction in actions)
        {
            agentAction.Initialize(envParams, modularAgentProps);
            actionTypes.AddRange(agentAction.GetActionTypes());
        }
        
        playerActionsManager.Initialize(actionTypes);
    }

    private void MaskList<T>(List<T> inputList, int bitMask)
    {
        int maxNumActionTypes = (int) Mathf.Log10(bitMask);
        int totalInputSize = inputList.Count;
        for (int i = totalInputSize - 1; i >= 0; i--)
        {
            int pow = maxNumActionTypes - i;
            if (pow >= 0)
            {
                int denominator = (int) Mathf.Pow(10, pow);
                int idx = (bitMask / denominator) % 2;
                if(idx==1) continue;
            }
            inputList.Remove(inputList[i]);
        }
    }

    public void InitializeAgentProperties()
    {
        if (!modularAgentProps) return;
        defaultGravity = modularAgentProps.GRAVITY_SCALE * Physics.gravity.y;
        fallingMultiplier = modularAgentProps.GRAVITY_FALL_MULTIPLIER;
        drag = modularAgentProps.MOVE_DRAG;
        dragModifier = 1.0f;
        velocityLimit = modularAgentProps.VELOCITY_LIMIT;
        actionRepetition = modularAgentProps.ACTION_REPETITION;

        groundCheckMask = modularAgentProps.GROUNDED_MASK;
        groundCheckOffset = modularAgentProps.GROUNDED_OFFSET;
        groundCheckRadius = modularAgentProps.GROUNDED_RADIUS;

        vectorNormalizationType = modularAgentProps.VECTOR_NORMALIZATION_TYPE;
        observedRotationType = modularAgentProps.OBSERVED_ROTATION_TYPE;
        vectorSpatialOrientationType = modularAgentProps.VECTOR_SPATIAL_ORIENTATION;
    }

    public void InitializeEnvironmentParameters()
    {
        if (envParams==null) return;
        observeVel = envParams.GetWithDefault("observe_velocity", observeVel ? 1 : 0) > 0;
        observeAccel = envParams.GetWithDefault("observe_acceleration", observeAccel ? 1 : 0) > 0;
        observeMutator = envParams.GetWithDefault("observe_mutator", observeMutator ? 1 : 0) > 0;
        actionRepetition = (int) envParams.GetWithDefault("num_action_repeat", actionRepetition);
        actionBitMask = (int) envParams.GetWithDefault("action_bit_mask", actionBitMask);
        
        vectorNormalizationType =
            (VectorNormalizationType) (int) envParams.GetWithDefault("vector_normalization_type", (int) vectorNormalizationType);
        observedRotationType = (ObservedRotationType) envParams.GetWithDefault("observed_rotation_type", (int) observedRotationType);
        vectorSpatialOrientationType =
            (VectorSpatialOrientationType) envParams.GetWithDefault("vector_spatial_orientation_type",
                (int) vectorSpatialOrientationType);
    }
    
    public void ProcessActions(ActionSegment<float> act)
    {
        // loop through each modular action component of agent, and send relevent action data.
        int i = 0;
        foreach (var action in actions)
        {
            if (i >= act.Length) continue;
            if (action.inputSize == 2) action.ProcessActionInput(act[i], act[i+1]);
            else action.ProcessActionInput(act[i]);
            i += action.inputSize;
        
            if(canTakeActions) action.Perform();
        }

        ApplyGravity();
        ApplyGroundVelocity();
        ApplyDrag();
    }


    private void FixedUpdate()
    {
        HandleGround();
        LimitVelocity();
    }

    private void LimitVelocity()
    {
        if (Velocity.sqrMagnitude > velocityLimit * velocityLimit)
        {
            agentRigidbody.velocity = Velocity.normalized * velocityLimit;
        }
    }

    private void ApplyGravity()
    {
        if (!gravityEnabled) return;
        Vector3 gravityDir = IsGrounded ? groundNormal : Vector3.up;
        float gravity = Velocity.y < 0 ? Gravity * fallingMultiplier : Gravity;
        agentRigidbody.AddForce(gravity*gravityDir, ForceMode.Acceleration);
    }
    
    private void ApplyDrag()
    {
        var magnitude = Drag;
        // if (!IsGrounded) magnitude *= AirControl;
        var dragForce = -Mathf.Abs(magnitude) * agentRigidbody.velocity;
        agentRigidbody.AddForce(dragForce, ForceMode.Acceleration);
    }

    private void ApplyGroundVelocity()
    {
        if (!grounding) return;
        Vector3 deltaPos = grounding.Velocity * Time.fixedDeltaTime;
        agentRigidbody.MovePosition(agentRigidbody.position + deltaPos);
        // tried to do with a change to velocity, but it either acts like a continuous force via VelocityChange, or does nothing if *Time.fixedDeltaTime.
        // i.e. agentRigidbody.velocity += grounding.Velocity OR agentRigidbody.AddForce(grounding.Velocity, ForceMode.VelocityChange)
        // Currently does not allow for carrying of momentum
    }

    private void HandleGround()
    {
        SetIsGrounded();
        SetGroundNormal();
    }

    private void SetIsGrounded()
    {
        if (!checkForGround)
        {
            isGrounded = false;
            return;
        }
        Vector3 offset = -transform.up * groundCheckOffset;
        isGrounded = Physics.CheckSphere(baseTransform.position - offset, groundCheckRadius, groundCheckMask,
            QueryTriggerInteraction.Ignore);
        
    }
    
    private void SetGroundNormal()
    {
        if (!IsGrounded)
        {
            groundNormal = Vector3.up;
            return;
        }
        // Use raycast to determine the normal of the ground to adjust the agent gravity and transform.
        var dir = -transform.up;
        var dist = groundCheckRadius;
        var offset = groundCheckOffset;
        Physics.Raycast(baseTransform.position - dir*offset, dir, out var hit, dist,
            groundCheckMask, QueryTriggerInteraction.Ignore);
        groundNormal = hit.collider ? hit.normal : Vector3.up;
    }

    public List<float> CollectObservations()
    {
        List<float> obs = new List<float>();
        
        // Forward vector y-component = sin(euler.x) whereas Pitch = euler.x/90f.
        // One is sinusoidal normalization vs linear normalization

        switch (observedRotationType)
        {
            case ObservedRotationType.YawPitch:
                obs.Add(Yaw); // (1)
                obs.Add(Pitch); // (1)
                break;
            case ObservedRotationType.Quaternion:
                obs.AddRange(AgentUtility.QuaternionToList(Rotation)); // (4)
                break;
            case ObservedRotationType.ForwardVector:
                obs.AddRange(AgentUtility.VectorToList(ForwardVector)); // (3)
                break;
        }
        
        if (observeVel)
        {
            var velocity = vectorSpatialOrientationType == VectorSpatialOrientationType.LocalSpace
                ? RelativeVelocity
                : Velocity;
            
            switch (vectorNormalizationType)
            {
                case VectorNormalizationType.Componentwise:
                    obs.AddRange(AgentUtility.VectorToList(velocity/velocityLimit)); // (3)
                    break;
                case VectorNormalizationType.DirectionMagnitude:
                    obs.AddRange(AgentUtility.GetNormalizedVectorObservation(velocity,velocityLimit,vectorNormalizationType)); // (4)
                    break;
            }
        }

        if (observeAccel)
        {
            var acceleration = vectorSpatialOrientationType == VectorSpatialOrientationType.LocalSpace
                ? RelativeAcceleration
                : Acceleration;
            
            switch (vectorNormalizationType)
            {
                case VectorNormalizationType.Componentwise:
                    obs.AddRange(AgentUtility.VectorToList(acceleration/AccelLimit)); // (3)
                    break;
                case VectorNormalizationType.DirectionMagnitude:
                    obs.AddRange(AgentUtility.GetNormalizedVectorObservation(acceleration,AccelLimit,vectorNormalizationType)); // (4)
                    break;
            }
        }
        
        // Observe if grounded (1 or 0)
        obs.Add(Convert.ToSingle(IsGrounded)); // (1)
        
        
        foreach (var action in actions)
        {
            obs.AddRange(action.CollectObservations());
        }

        if (observeMutator)
        {
            var oneHot = new float[Enum.GetValues(typeof(MutatorType)).Length];
            if (currentMutatorType != MutatorType.None)
            {
                int idx = (int) Mathf.Log((float) currentMutatorType, 2f);
                if(idx>=0 && idx<oneHot.Length) oneHot[idx] = 1;
            }
            obs.AddRange(oneHot);
        }

        oldVelocity = Velocity;

        return obs;
    }

    public int GetNumMovementObservations()
    {
        return CollectObservations().Count;
    }

    public void SetRotation(Quaternion rotation)
    {
        rotationQuaternion = rotation;
        var euler = rotationQuaternion.eulerAngles;
        if (rotateTransform) transform.rotation = Quaternion.Euler(0f, euler.y, 0f);
        cameraFollowTransform.rotation = rotationQuaternion;
    }

    public void SetPositionAndRotation(Vector3 position, Quaternion rotation)
    {
        var pos = position + 1f*Vector3.up;
        transform.position = pos;
        agentRigidbody.position = pos;
        
        ResetRigidbody();

        SetRotation(rotation);
    }

    private void ResetRigidbody()
    {
        agentRigidbody.velocity = Vector3.zero;
        agentRigidbody.ResetInertiaTensor();
        agentRigidbody.ResetCenterOfMass();
    }


    

    public void Reset()
    {
        canTakeActions = true;
        gravityEnabled = true;
        rotateTransform = true;

        foreach (var action in actions)
        {
            action.Reset();
        }
        
        ResetRigidbody();
        
        HandleGround();
    }

    public Dictionary<ComplexActionType, List<Vector3>> GetComplexActionPositions()
    {
        var complexActionPositions = new Dictionary<ComplexActionType, List<Vector3>>();
        foreach (var action in actions)
        {
            var type = action.complexActionType;
            if(type==ComplexActionType.Null) continue;
            complexActionPositions.Add(type,action.GetComplexActionPositions());
        }

        return complexActionPositions;
    }

    // private void OnDrawGizmos()
    // {
    //     Gizmos.color = Color.red;
    //     Vector3 offset = -transform.up * groundCheckOffset;
    //     Gizmos.DrawWireSphere(baseTransform.position - offset, groundCheckRadius);
    //     Gizmos.DrawRay(baseTransform.position - offset, -transform.up*groundCheckRadius);
    // }
}
