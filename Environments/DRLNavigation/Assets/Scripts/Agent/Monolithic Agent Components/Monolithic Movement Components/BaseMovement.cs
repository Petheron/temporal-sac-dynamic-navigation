using UnityEngine;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Quaternion = UnityEngine.Quaternion;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;


public class BaseMovement:MonoBehaviour
{
    [Header("Movement Behaviour Toggles")]
    public bool usePitch = true;
    public bool useMomentum = true;

    [SerializeField] private bool observeVel = true;
    [SerializeField] private bool observeAccel = true;

    protected int numObs = 0;
    
    [Header("Action Buffer Indices")] 
    [SerializeField] protected int forwardIndex = 0;
    [SerializeField] protected int strafeIndex = 1;
    [SerializeField] protected int yawIndex = 2;
    [SerializeField] protected int pitchIndex = 3;
    [SerializeField] protected int jumpIndex = 4;
    
    [Header("Player")]
    [Tooltip("Forward Move speed of the character in m/s")]
    public float defaultForwardSpeed = 60f;
    [Tooltip("Strafe speed of the character in m/s")]
    public float defaultStrafeSpeed = 60f;
    [Tooltip("Modifier to movement inputs to horizontal velocity in the air.")]
    public float airSpeedModifier = 0.5f;
    [Tooltip("Rotation speed of the character")]
    public float lookSensitivity = 5.0f;
    [Tooltip("Minimum square change for look delta.")]
    public float minimumLookThreshold = 0.0001f;
    [Tooltip("Gravity for player")]
    public float defaultGravity = -19.62f;
    protected float m_Gravity;
    [Tooltip("Player drag while touching the ground")]
    public float groundDrag = 5f;
    [Tooltip("Player drag while touching the ground")]
    public float airDrag = 1f;
    [Header("Current Movement Values")]
    [SerializeField]protected float forwardSpeed;
    [SerializeField]protected float strafeSpeed;
    protected float AirSpeedLimit => forwardSpeed;
    [SerializeField] protected float airSpeedDecayModifier = 1.0f;

    [Space(10)]
    [Header("Grounded Parameters")]
    [Tooltip("Transform representing base of agent. Used for goal condition & ground checks.")]
    public Transform baseTransform;
    [SerializeField, ReadOnly] private bool m_IsGrounded = false;
    private float m_GroundCheckOffset = 0.41f;
    private float m_GroundCheckRadius = 0.44f;
    [SerializeField] protected LayerMask m_GroundCheckMask;
    protected Vector3 GroundNormal = Vector3.up;
    public float cachedGroundVelocityDecayTime = 0.5f;
    [SerializeField,ReadOnly] private float cachedGroundVelocityDecayTimer;


    [Space(10)]
    [Header("Cinemachine")]
    [Tooltip("The follow target set in the Cinemachine Virtual Camera that the camera will follow")]
    public Transform cinemachineFollowTargetTransform;
    [SerializeField] protected Vector3 originalCameraPosition;
    [Tooltip("How far in degrees can you move the camera up")]
    private float topAngleClamp = 90.0f;
    [Tooltip("How far in degrees can you move the camera down")]
    private float bottomAngleClamp = -90.0f;
    [Tooltip("Does the agent transform align with the camera rotation. Disable to look around without affecting movement.")]
    protected bool isRotateTransform = true;

    [Space(10)] [Header("Observation Normalization Values")]
    public int actionRepeatRate = 1;
    [SerializeField] protected float velocityLimit = 100f;
    
    protected Rigidbody agentRigidbody;
    protected Transform agentTransform;
    protected bool canApplyMoveInput = true;
    protected Vector3 slopedRightVector;
    protected Vector3 slopedForwardVector;
    private Vector3 oldVelocity;

    [SerializeField] protected Rigidbody groundRigidbody;
    private bool isPlatformVelocityPreserved = false;
    
    [SerializeField, ReadOnly] private Vector3 rbVelocity;
    [SerializeField,ReadOnly] private Vector3 cachedGroundVelocity = Vector3.zero;
    [SerializeField,ReadOnly] private Vector3 initialCachedGroundVelocity = Vector3.zero;

    public Dictionary<ComplexActionType, List<Vector3>> complexActionPositions;

    public Vector3 Position => baseTransform.position;
    
    private float yaw = 0f;
    private float Pitch
    {
        get
        {
            float p = cinemachineFollowTargetTransform.rotation.eulerAngles.x;
            p = p > 180f ? p - 360f : p;
            return p/90f;
        }
    }

    private float Yaw
    {
        get
        {
            float y = yaw;
            y = Mathf.Repeat(y, 360f);
            return y/180f -1;
        }
    }

    protected Vector3 Velocity
    {
        get => agentRigidbody.velocity;
        set => agentRigidbody.velocity = value;
    }
    
    protected Vector3 Acceleration => (Velocity - oldVelocity) / (actionRepeatRate*Time.fixedDeltaTime);
    private float AccelLimit => 2*velocityLimit/(Time.fixedDeltaTime);

    protected bool IsGrounded => m_IsGrounded;
    
    

    protected virtual void Awake()
    {
        agentRigidbody = GetComponent<Rigidbody>();
        agentTransform = transform;
        
        forwardSpeed = defaultForwardSpeed;
        strafeSpeed = defaultStrafeSpeed;
        m_Gravity = defaultGravity;

        complexActionPositions = new Dictionary<ComplexActionType, List<Vector3>>();
    }

    protected virtual void Start()
    {
        if (agentRigidbody)
        {
            agentRigidbody.useGravity = false;
            agentRigidbody.freezeRotation = true;
            agentRigidbody.angularDrag = 0f;
        }

        if (cinemachineFollowTargetTransform) cinemachineFollowTargetTransform.localPosition = originalCameraPosition;
    }

    public virtual void SetDefaultMovementProperties(MonolithicAgentProperties agentProperties)
    {
        if (!agentProperties) return;

        if (useMomentum)
        {
            defaultForwardSpeed = agentProperties.MOVE_FORCE;
            defaultStrafeSpeed = agentProperties.MOVE_FORCE;
            groundDrag = agentProperties.GROUND_DRAG;
            airDrag = agentProperties.AIR_DRAG;
            airSpeedModifier = agentProperties.AIR_SPEED_MODIFIER;
        }
        else
        {
            defaultForwardSpeed = agentProperties.MOVE_SPEED;
            defaultStrafeSpeed = agentProperties.MOVE_SPEED;
            groundDrag = 0f;
            airDrag = 0f;
            airSpeedModifier = 1.0f;
            airSpeedDecayModifier = agentProperties.AIR_SPEED_DECAY_MODIFIER;
        }
        
        lookSensitivity = agentProperties.ROTATE_SPEED;
        
        
        defaultGravity = agentProperties.GRAVITY_SCALE * Physics.gravity.y;
        velocityLimit = agentProperties.VELOCITY_LIMIT;

        m_GroundCheckOffset = agentProperties.GROUNDED_OFFSET;
        m_GroundCheckRadius = agentProperties.GROUNDED_RADIUS;
        m_GroundCheckMask = agentProperties.GROUNDED_MASK;
        
        forwardSpeed = defaultForwardSpeed;
        strafeSpeed = defaultStrafeSpeed;
        m_Gravity = defaultGravity;
        
        if (!usePitch)
        {
            pitchIndex = -1;
            jumpIndex -= 1;
        }
    }

    public virtual void ReceiveEnvironmentParameters(EnvironmentParameters parameters)
    {
        useMomentum = parameters.GetWithDefault("use_momentum", useMomentum?1:0)>0;
        usePitch = parameters.GetWithDefault("use_pitch", usePitch?1:0)>0;
        observeVel = parameters.GetWithDefault("observe_velocity", observeVel ? 1 : 0) > 0;
        observeAccel = parameters.GetWithDefault("observe_acceleration", observeAccel ? 1 : 0) > 0;
    }

    public virtual int SetNumObs()
    {
        int o = 0;
        o += 3; // for yaw, pitch, isGrounded
        if (observeVel) o += 3;
        if (observeAccel) o += 3;
        numObs = o;
        return o;
    }

    protected virtual void FixedUpdate()
    {
        SetIsGrounded();
        SetDrag(IsGrounded?groundDrag:airDrag);
        SetGroundNormal();
        DecrementTimers();
    }

    private void DecayCachedGroundVelocity()
    {
        if (groundRigidbody) return;

        if (initialCachedGroundVelocity.y < 0f)
            initialCachedGroundVelocity = new Vector3(initialCachedGroundVelocity.x, 0f, initialCachedGroundVelocity.z);
        
        cachedGroundVelocity = (cachedGroundVelocityDecayTimer / (cachedGroundVelocityDecayTime+Mathf.Epsilon)) *
                               initialCachedGroundVelocity;
        
        if (cachedGroundVelocityDecayTimer <= 0f) initialCachedGroundVelocity = Vector3.zero;
        
        if (initialCachedGroundVelocity.y > 0f)
            initialCachedGroundVelocity = new Vector3(initialCachedGroundVelocity.x, 0f, initialCachedGroundVelocity.z);
        
        

    }

    private void StoreOldVelocity()
    {
        oldVelocity = Velocity;
    }

    protected void LateUpdate()
    {
        if (Velocity.sqrMagnitude > velocityLimit * velocityLimit)
        {
            agentRigidbody.velocity = Velocity.normalized * velocityLimit;
        }
        rbVelocity = Velocity;
    }

    /// <summary>
    /// Perform any timer decrementation. Called during FixedUpdate. 
    /// </summary>
    protected virtual void DecrementTimers()
    {
        cachedGroundVelocityDecayTimer = cachedGroundVelocityDecayTimer > 0 ? cachedGroundVelocityDecayTimer - Time.fixedDeltaTime : 0f;
    }

    /// <summary>
    /// Update the drag value of the agent rigidbody. 
    /// </summary>
    /// <param name="newDrag"></param>
    protected void SetDrag(float newDrag)
    {
        agentRigidbody.drag = newDrag;
    }

    /// <summary>
    /// Reset the movement of the agent for episode begin.
    /// </summary>
    /// <param name="position">Reset position.</param>
    /// <param name="rotation">Reset rotation</param>
    public virtual void ResetMovement(Vector3 position, Quaternion rotation)
    {
        oldVelocity = Vector3.zero;
        if (!agentTransform) agentTransform = transform;
        agentTransform.position = position;
        agentTransform.rotation = rotation;
        yaw = rotation.eulerAngles.y;
        
        if (!agentRigidbody) agentRigidbody = GetComponent<Rigidbody>();
        agentRigidbody.position = position;
        agentRigidbody.rotation = rotation;
        agentRigidbody.velocity = Vector3.zero;
        agentRigidbody.ResetInertiaTensor();
        agentRigidbody.ResetCenterOfMass();

        if (cinemachineFollowTargetTransform)
        {
            cinemachineFollowTargetTransform.localRotation = Quaternion.Euler(Vector3.zero);
        }

        forwardSpeed = defaultForwardSpeed;
        strafeSpeed = defaultStrafeSpeed;
        cachedGroundVelocityDecayTimer = 0f;
        initialCachedGroundVelocity = Vector3.zero;
        
        complexActionPositions.Clear();
    }

    /// <summary>
    /// Rotate transform and camera based on look inputs.
    /// Pitch change is clamped to prevent agent from looking more than 90 degrees up or down.
    /// </summary>
    /// <param name="yawInput">Input to rotate yaw.</param>
    /// <param name="pitchInput">Input to rotate pitch by.</param>
    protected void Look(float yawInput, float pitchInput)
    {
        if (yawInput*yawInput + pitchInput*pitchInput <= minimumLookThreshold) return;

        if (!usePitch) pitchInput = 0f;

        float oldLook = 5f;
        var yawDelta = yawInput * oldLook;
        var pitchDelta = -pitchInput * oldLook;
        
        // var yawDelta = yawInput * lookSensitivity * Time.fixedDeltaTime;
        // var pitchDelta = -pitchInput * lookSensitivity * Time.fixedDeltaTime;

        if (!cinemachineFollowTargetTransform)
            return;
        
        var pitch = cinemachineFollowTargetTransform.localRotation.eulerAngles.x + pitchDelta;
        pitch = ClampEulerAngle(pitch, bottomAngleClamp, topAngleClamp);

        yaw += yawDelta;
        
        // Perform rotation based on either transform (effects movement) or the camera (independent of movement)
        if (isRotateTransform)
        {
            cinemachineFollowTargetTransform.localRotation = Quaternion.Euler(pitch,0f,0f);
            agentTransform.rotation = Quaternion.Euler(0f,yaw,0f);
        }
        else
        {
            cinemachineFollowTargetTransform.rotation = Quaternion.Euler(pitch,yaw,0f);
        }
    }
    
    /// <summary>
    /// Process movement inputs and calculate the horizontal movement vectors according to the ground normal.
    /// </summary>
    /// <param name="strafeInput">Strafe input value between -1 and 1.</param>
    /// <param name="forwardInput">Forward input value between -1 and 1.</param>
    protected virtual void ProcessMoveInputs(float strafeInput, float forwardInput)
    {
        var strafe = strafeInput;
        var forward = forwardInput;
        var inputVector = new Vector2(strafe, forward);
        inputVector = inputVector.normalized;
        strafe = inputVector.x;
        forward = inputVector.y;

        // Accommodate for sloped surface.
        slopedRightVector = Vector3.Cross(-agentTransform.forward, GroundNormal)*strafe;
        slopedForwardVector = Vector3.Cross(agentTransform.right, GroundNormal)*forward;
    }

    /// <summary>
    /// Apply movement directions to rigidbody via mass-free acceleration.
    /// </summary>
    protected virtual void Move()
    {
        if (!canApplyMoveInput) return;
        if (useMomentum)
        {
            MomentumMove();
        }
        else
        {
            VelocityMove();
        }

        GroundMove();
        
    }

    private void MomentumMove()
    {
        Vector3 horizontalForce = slopedRightVector*strafeSpeed + slopedForwardVector*forwardSpeed;
        if (!IsGrounded) horizontalForce *= airSpeedModifier;
        agentRigidbody.AddForce(horizontalForce,ForceMode.Acceleration);
    }

    private void VelocityMove()
    {
        Vector3 horizontalVel = slopedRightVector*strafeSpeed + slopedForwardVector*forwardSpeed;
        if (!IsGrounded)
        {
            horizontalVel *= airSpeedModifier;
            // horizontalVel = Vector3.ProjectOnPlane(Velocity, GroundNormal) + horizontalVel;
            // if (horizontalVel.sqrMagnitude > AirSpeedLimit * AirSpeedLimit)
            // {
            //     var speed = horizontalVel.magnitude - airSpeedDecayModifier*AirSpeedLimit;
            //     horizontalVel = horizontalVel.normalized * speed;
            // }
        }
        var verticalVel = Vector3.Project(Velocity, GroundNormal);
        agentRigidbody.velocity = horizontalVel + verticalVel;
    }

    public void AttachGroundRigidbody(Rigidbody rb)
    {
        if (!rb)
        {
            cachedGroundVelocity = Vector3.zero;
        }
        groundRigidbody = rb;
        cachedGroundVelocityDecayTimer = 0f;
    }
    
    public void DetachGroundRigidBody(Rigidbody rigidbodyBeingLeft)
    {
        // If ground being left is the current ground, we are jumping, so detach.
        // This avoids lack of attachment when walking between two ground gameobjects
        if (rigidbodyBeingLeft == groundRigidbody)
        {
            groundRigidbody = null;
            cachedGroundVelocityDecayTimer = isPlatformVelocityPreserved ? cachedGroundVelocityDecayTime : 0f;
        }
    }
    
    private void GroundMove()
    {
        if (groundRigidbody)
        {
            var vel = groundRigidbody.velocity;
            cachedGroundVelocity = new Vector3(vel.x, 0f, vel.z);
            initialCachedGroundVelocity = vel;
        }
        else
        {
            DecayCachedGroundVelocity();
        }
        
        agentRigidbody.velocity += cachedGroundVelocity;
    }

    /// <summary>
    /// Applies gravity to the rigidbody ignoring mass. Gravity direction aligned with ground normal to prevent sliding down slopes.
    /// </summary>
    protected void ApplyGravity()
    {
        Vector3 gravityDir = m_IsGrounded ? GroundNormal : Vector3.up;
        var g = Velocity.y < 0 ? m_Gravity * 1.5f : m_Gravity;
        agentRigidbody.velocity += gravityDir * (g * Time.fixedDeltaTime);
    }

    /// <summary>
    /// Process the actions of the agent and apply to rigidbody.
    /// </summary>
    /// <param name="act">Actions of agent. Order [forward, strafe, yaw]</param>
    /// <param name="heuristicMouseScale">If heuristically-controlled, apply mouseScaling to revert mouse input normalization.</param>
    public virtual void ProcessActions(ActionSegment<float> act, float heuristicMouseScale=1f)
    {
        var forward = Mathf.Clamp(act[forwardIndex], -1.0f, 1.0f);
        var strafe = Mathf.Clamp(act[strafeIndex], -1.0f, 1.0f);
        var yaw = Mathf.Clamp(act[yawIndex], -1.0f, 1.0f);
        var pitch = 0f;
        if(usePitch) pitch = Mathf.Clamp(act[pitchIndex], -1.0f, 1.0f);
        
        ProcessMoveInputs(strafe,forward);
        Move();
        ApplyGravity();
        Look(yaw,pitch);
    }

    /// <summary>
    /// Returns a list of normalized float observations.
    /// Observations in order [velocity, acceleration, isGrounded, YawRotation, PitchRotation] [9 observations].
    /// </summary>
    /// <returns></returns>
    public virtual List<float> CollectVectorObservations()
    {
        var obs = new List<float>();
        var vel = new []{
            Velocity.x/velocityLimit, 
            Velocity.y/velocityLimit, 
            Velocity.z/velocityLimit
        };
        var accel = new []{
            Acceleration.x/AccelLimit, 
            Acceleration.y/AccelLimit, 
            Acceleration.z/AccelLimit
        };
        // print($"vel: {Velocity}, old_vel: {oldVelocity}, Accel: {Acceleration} ");
        if(observeVel) obs.AddRange(vel); // 3
        if(observeAccel) obs.AddRange(accel); // 3
        obs.Add(IsGrounded?1f:0f); // 1
        obs.Add(Yaw); // 1
        obs.Add(Pitch); // 1
        StoreOldVelocity();
        return obs;
    }
    
    /// <summary>
    /// Set IsGrounded based on ground contact.
    /// </summary>
    protected void SetIsGrounded()
    {
        // Uses CheckSphere to determine if the agent is grounded.
        var dir = -agentTransform.up;
        var offset = m_GroundCheckOffset;
        m_IsGrounded = Physics.CheckSphere(baseTransform.position - dir*offset, m_GroundCheckRadius, m_GroundCheckMask,
            QueryTriggerInteraction.Ignore);
    }
    
    /// <summary>
    /// Set ground normal if on ground.
    /// </summary>
    protected void SetGroundNormal()
    {
        if (!m_IsGrounded)
        {
            GroundNormal = Vector3.up;
            return;
        }
        // Use raycast to determine the normal of the ground to adjust the agent gravity and transform.
        var dir = -agentTransform.up;
        var dist = m_GroundCheckRadius;
        var offset = m_GroundCheckOffset;
        Physics.Raycast(baseTransform.position - dir*offset, dir, out var hit, dist,
            m_GroundCheckMask, QueryTriggerInteraction.Ignore);
        GroundNormal = hit.collider ? hit.normal : Vector3.up;
    }
    
    /// <summary>
    /// Clamp pitch rotation between lfMin and lfMax degrees.
    /// Use for euler angle of rotation as accommodates for 360 flip from 1 -> 359.
    /// </summary>
    /// <param name="lfAngle">Angle to clamp</param>
    /// <param name="lfMin">Minimum angle</param>
    /// <param name="lfMax">Maximum angle</param>
    /// <returns></returns>
    protected static float ClampEulerAngle(float lfAngle, float lfMin, float lfMax)
    {
        if (lfAngle > 180) lfAngle -= 360f;
        return Mathf.Clamp(lfAngle, lfMin, lfMax);
    }


}
