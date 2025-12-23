using System;
using System.Collections.Generic;
using Cinemachine;
using Unity.MLAgents;
using UnityEngine;
using Unity.MLAgents.Actuators;

public class FPSBaseMovement:JumpMovement
{
    [Header("Sprinting")] 
    public bool useSprint = true;
    public bool isSprintToggle = false;
    public float sprintModifier = 1.3f;
    public bool isSprinting = false;
    public float minimumSprintDuration = 0.2f;
    private float sprintTimer;
    private bool CanSprint => !(sprintTimer > 0f);
    
    [Header("Crouching")]
    public bool useCrouch = true;
    public float crouchModifier = 0.5f;
    public bool isCrouching = false;
    public float crouchLerpDuration = 0.1f;
    private float crouchLerpTimer = 0f;
    public float crouchHeightOffset = 0.5f;
    [Header("Sliding")] 
    public bool useSlide = true;
    private bool isSliding = false;
    public float slideForce = 10f;
    public float slideSpeed = 10f;
    public float slideDuration = 0.2f;
    private float slideTimer = 0f;
    [SerializeField] private Vector3 slideDirection;

    private CapsuleCollider agentCollider;
    private float m_CapsuleHeight = 2f;
    private Vector3 m_CapsuleCenter = Vector3.zero;
    
    [Header("Wall Riding")] 
    public bool useWallRide = true;
    [SerializeField] private bool isWallRide = false;
    [Space(10)] 
    public float wallrideGravityModifier = 0.5f;
    public float wallrideDragModifier = 0.5f;
    public float wallCheckDistance = 0.55f;
    public float wallVerticalityMinimum = 0.1f;
    public float wallRideVelocityTowardWallMax = 0.1f;
    [Space(10)] 
    protected bool isWallJumping = false;
    protected Vector3 wallJumpDirection;
    protected float wallJumpTimer = 0f;
    public float wallJumpForce = 20f;
    public float jumpForceModifierForWallJump = 0.1f;
    public float wallJumpVelocity = 12.5f;
    public float wallJumpDuration = 0.25f;

    [Space(10)] 
    public CinemachineVirtualCamera CinemachineVirtualCamera;
    public float wallRideDutchAngle = 15.0f;
    public float wallRideCameraLerpTimer;
    public float wallRideCameraLerpDuration = 0.1f;

    [SerializeField]private Vector3 wallNormal;
    private Vector3 wallRideDirection;

    protected override void Awake()
    {
        base.Awake();
        if (!agentCollider)
        {
            agentCollider = GetComponent<CapsuleCollider>();
        }
        if (agentCollider)
        {
            m_CapsuleHeight = agentCollider.height;
            m_CapsuleCenter = agentCollider.center;
        }
    }

    public override void ResetMovement(Vector3 position, Quaternion rotation)
    {
        base.ResetMovement(position, rotation);
        isSprinting = false;
        isCrouching = false;
        crouchLerpTimer = 0f;
        slideTimer = 0f;
        isSliding = false;
        wallRideCameraLerpTimer = 0f;
        wallNormal = Vector3.zero;
        UpdateCameraPosition();
        UpdateCameraTilt();
        if (agentCollider)
        {
            agentCollider.height = m_CapsuleHeight;
            agentCollider.center = m_CapsuleCenter;
        }
        
        complexActionPositions[ComplexActionType.Sprint] = new List<Vector3>();
        complexActionPositions[ComplexActionType.Crouch] = new List<Vector3>();
        complexActionPositions[ComplexActionType.WallRide] = new List<Vector3>();
    }

    public override void SetDefaultMovementProperties(AgentProperties agentProperties)
    {
        base.SetDefaultMovementProperties(agentProperties);

        sprintModifier = agentProperties.SPRINT_MODIFIER;
        
        crouchModifier = agentProperties.CROUCH_SPEED_MODIFIER;
        crouchLerpDuration = agentProperties.CROUCH_LERP_DURATION;
        crouchHeightOffset = agentProperties.CROUCH_HEIGHT_OFFSET;

        slideForce = agentProperties.SLIDE_FORCE;
        slideSpeed = agentProperties.SLIDE_SPEED;
        slideDuration = agentProperties.SLIDE_DURATION;

        wallJumpForce = agentProperties.WALLRIDE_JUMP_FORCE;
        jumpForceModifierForWallJump = agentProperties.WALLRIDE_JUMP_FORCE_MODIFIER;
        
        wallJumpVelocity = agentProperties.WALLJUMP_VELOCITY;
        wallJumpDuration = agentProperties.WALLJUMP_DURATION;

        wallrideGravityModifier = agentProperties.WALLRIDE_GRAVITY_MODIFIER;
        wallrideDragModifier = agentProperties.WALLRIDE_DRAG_MODIFIER;

        wallCheckDistance = agentProperties.WALLRIDE_CHECK_DISTANCE;
        wallVerticalityMinimum = agentProperties.WALLRIDE_VERTICALITY_MINIMUM;
        wallRideVelocityTowardWallMax = agentProperties.WALLRIDE_VELOCITY_TOWARDS_WALL_MAX;

        wallRideDutchAngle = agentProperties.WALLRIDE_DUTCH_ANGLE;
        wallRideCameraLerpDuration = agentProperties.WALLRIDE_CAMERA_LERP_DURATION;
    }

    public override void ReceiveEnvironmentParameters(EnvironmentParameters parameters)
    {
        base.ReceiveEnvironmentParameters(parameters);

        useSprint = parameters.GetWithDefault("use_sprint", useSprint?1:0) > 0;
        useCrouch = parameters.GetWithDefault("use_crouch", useCrouch?1:0) > 0;
        useSlide = parameters.GetWithDefault("use_slide", useSlide?1:0) > 0;
        useWallRide = parameters.GetWithDefault("use_wallride", useWallRide?1:0) > 0;
    }

    public override int SetNumObs()
    {
        int o =  base.SetNumObs() + 2;
        if (useSlide) o += 1;
        if (useWallRide) o += 1;
        numObs = o;
        return o;
    }

    protected override void DecrementTimers()
    {
        base.DecrementTimers();

        if (sprintTimer > 0)
        {
            sprintTimer -= Time.fixedDeltaTime;
            if (sprintTimer < 0) sprintTimer = 0;
        }
        
        if (crouchLerpTimer > 0)
        {
            crouchLerpTimer -= Time.fixedDeltaTime;
            if (crouchLerpTimer <= 0)
            {
                crouchLerpTimer = 0f;
            }
            UpdateCameraPosition();
        }
        
        if (slideTimer > 0)
        {
            slideTimer -= Time.fixedDeltaTime;
            if (slideTimer <= 0)
            {
                slideTimer = 0f;
                isSliding = false;
            }
        }

        if (!useMomentum)
        {
            if (wallJumpTimer > 0)
            {
                wallJumpTimer -= Time.fixedDeltaTime;
                if (wallJumpTimer <= 0f)
                {
                    CancelWallJumpVelocity();
                }
            }
        }

        if (wallRideCameraLerpTimer>0f)
        {
            wallRideCameraLerpTimer -= Time.fixedDeltaTime;
            if (wallRideCameraLerpTimer <= 0)
            {
                wallRideCameraLerpTimer = 0f;
            }
            UpdateCameraTilt();
        }
        else
        {
            if (IsGrounded && wallNormal != Vector3.zero) wallNormal = Vector3.zero;
        }
    }

    /// <summary>
    /// Lerp the camera position depending on crouch timer. Also adjusts capsule height & centre to allow for ducking under things.
    /// </summary>
    private void UpdateCameraPosition()
    {
        var t = isCrouching? 1.0f - crouchLerpTimer / crouchLerpDuration : crouchLerpTimer / crouchLerpDuration;
        cinemachineFollowTargetTransform.localPosition = Vector3.Lerp(originalCameraPosition, originalCameraPosition-crouchHeightOffset*Vector3.up, t);
        if (agentCollider)
        {
            agentCollider.center = Vector3.Lerp(m_CapsuleCenter, m_CapsuleCenter-0.5f*crouchHeightOffset*Vector3.up, t);
            agentCollider.height = Mathf.Lerp(m_CapsuleHeight, m_CapsuleHeight - crouchHeightOffset, t);
        }
    }

    /// <summary>
    /// Tilts the camera away from 0 if wallriding, otherwise moves towards zero. Direction of tilt determined by side the wall is on.
    /// Did not use lerp here as it would flicker on wall jump.
    /// </summary>
    private void UpdateCameraTilt()
    {
        if (!CinemachineVirtualCamera) return;
        var endAngle = 0f;
        var deltaAnglePerFrame = wallRideDutchAngle * Time.fixedDeltaTime / wallRideCameraLerpDuration;
        if (isWallRide)
        {
            var dot = Vector3.Dot(wallNormal, -transform.right);
            endAngle = wallRideDutchAngle * Mathf.Sign(dot);
        }
        var currentAngle = CinemachineVirtualCamera.m_Lens.Dutch;
        var newAngle = currentAngle;
        if (currentAngle < endAngle) newAngle += deltaAnglePerFrame;
        else if (currentAngle > endAngle) newAngle -= deltaAnglePerFrame;

        if (isWallRide)
        {
            if (Mathf.Abs(newAngle) > wallRideDutchAngle) newAngle = Mathf.Sign(newAngle) * wallRideDutchAngle;
        }
        else
        {
            if (Mathf.Sign(currentAngle*newAngle).Equals(-1) && newAngle != 0f && currentAngle != 0f)
            {
                newAngle = 0f;
            }
        }
        
        if (wallRideCameraLerpTimer == 0)
        {
            newAngle = endAngle;
            if (IsGrounded)
            {
                wallNormal = Vector3.zero;
            }
        }
        
        CinemachineVirtualCamera.m_Lens.Dutch = newAngle;
    }
    
    /// <summary>
    /// Change sprint state based to new SprintState if possible.
    /// Cannot sprint if not grounded. 
    /// </summary>
    /// <param name="newSprintState"></param>
    private void ProcessSprint(bool newSprintState)
    {
        if (!IsGrounded) return;
        if (!useSprint) return;
        if (!CanSprint) return;
        
        // flip sprint state if toggle is on
        if (isSprintToggle)
        {
            if (!newSprintState) return;
            if(isSprinting) EndSprint();
            else StartSprint();
            return;
        }

        // Handle hold sprinting
        if(newSprintState && !isSprinting) StartSprint();
        else if (!newSprintState && isSprinting) EndSprint();
    }
    
    /// <summary>
    /// Begin sprinting. Cannot sprint if crouching.
    /// </summary>
    private void StartSprint()
    {
        if (isCrouching) return;
        isSprinting = true;
        forwardSpeed = sprintModifier*defaultForwardSpeed;
        complexActionPositions[ComplexActionType.Sprint].Add(transform.position);
        sprintTimer = minimumSprintDuration;
    }
    
    /// <summary>
    /// Cancel sprinting. Restores forward speed to either walking or crouching speed.
    /// </summary>
    private void EndSprint()
    {
        isSprinting = false;
        forwardSpeed = !isCrouching? defaultForwardSpeed: crouchModifier*defaultForwardSpeed;
        sprintTimer = minimumSprintDuration;
    }

    /// <summary>
    /// Change crouch state if possible.
    /// Prevents crouching if in air, mid-crouch transition, or sliding.
    /// </summary>
    /// <param name="newCrouchState"></param>
    private void ProcessCrouch(bool newCrouchState)
    {
        if (!useCrouch) return;
        if (!IsGrounded) return;
        if (crouchLerpTimer > 0) return;
        if (slideTimer > 0) return;
        
        switch (newCrouchState)
        {
            case true when !isCrouching:
                StartCrouch();
                break;
            case false when isCrouching:
                EndCrouch();
                break;
        }
    }

    /// <summary>
    /// Begin crouching.
    /// </summary>
    private void StartCrouch()
    {
        isCrouching = true;
        crouchLerpTimer = crouchLerpDuration;
        // If sprinting, stop sprinting and slide.
        if (isSprinting)
        {
            EndSprint();
            StartSlide();
        }
        // slow down to crouch speed. Also minimizes effect on slide.
        forwardSpeed = defaultForwardSpeed*crouchModifier;
        strafeSpeed = defaultStrafeSpeed*crouchModifier;
        complexActionPositions[ComplexActionType.Crouch].Add(transform.position);
    }

    /// <summary>
    /// End crouch state & restore movement to normal speed.
    /// </summary>
    private void EndCrouch()
    {
        isCrouching = false;
        crouchLerpTimer = crouchLerpDuration;
        slideTimer = 0f;
        isSliding = false;
        forwardSpeed = defaultForwardSpeed;
        strafeSpeed = defaultStrafeSpeed;
    }
    
    /// <summary>
    /// Begin sliding. Starts sliding timer and provides initial thrust.
    /// </summary>
    private void StartSlide()
    {
        if (!useSlide) return;
        if (isSliding) return;
        // Slide, add boost to speed/apply one off force
        slideTimer = slideDuration;
        isSliding = true;
        if (useMomentum)
        {
            agentRigidbody.AddForce((slopedRightVector+slopedForwardVector)*slideForce, ForceMode.VelocityChange);
        }
        else
        {
            slideDirection = slopedForwardVector + slopedRightVector;
        }
    }

    private void Slide()
    {
        if (useMomentum) return;
        if (!isSliding) return;
        var speed = Mathf.Lerp(0, slideSpeed, slideTimer / slideDuration);
        agentRigidbody.velocity = slideDirection * speed +
                                  Vector3.Project(agentRigidbody.velocity, GroundNormal);
    }
    
    /// <summary>
    /// If possible, jump with the provided force. Also applies walljump if wallriding.
    /// </summary>
    /// <param name="isJumpInput"></param>
    /// <param name="force"></param>

    protected override void Jump(bool isJumpInput, float force, float multiplier=1.0f)
    {
        if (!isJumpInput) return;
        // Uncrouch if necessary.
        if(isCrouching) EndCrouch();
        // Adjust vertical jump if wallriding.

        if (isWallRide)
        {
            remainingMidairJumps++;
            multiplier = jumpForceModifierForWallJump;
        }
        
        
        base.Jump(isJumpInput, force, multiplier);
        
        if (isWallRide)
        {
            WallJump();
        }
    }

    /// <summary>
    /// Perform walljump if wallriding. Applies force in direction of wall normal and resets jumping capabilities.
    /// </summary>
    private void WallJump()
    {
        if (!isWallRide) return;

        if (useMomentum)
        {
            agentRigidbody.AddForce(wallNormal*wallJumpForce,ForceMode.VelocityChange);
        }
        else
        {
            isWallJumping = true;
            wallJumpDirection = wallNormal;
            wallJumpTimer = wallJumpDuration;
            // agentRigidbody.velocity += wallNormal * wallJumpForce;
        }

        CancelWallRide();
        // canJump = true;
    }

    private void ApplyWallJumpVelocity()
    {
        if (useMomentum) return;
        if (!isWallJumping) return;
        var speed = Mathf.Lerp(0, wallJumpVelocity, wallJumpTimer / wallJumpDuration);
        agentRigidbody.velocity += wallJumpDirection * speed;
    }

    private void CancelWallJumpVelocity()
    {
        wallJumpTimer = 0f;
        wallJumpDirection = Vector3.zero;
        isWallJumping = false;
    }

    private void GetWallNormalForWallRide(float strafeInput, float forwardInput)
    {
        if (IsGrounded) return;
        RaycastHit hit = new RaycastHit();
        var t = transform;
        var velDir = Vector3.ProjectOnPlane(Velocity, t.up).normalized;
        var isVelDirWall = Physics.Raycast(t.position, velDir, out var hitVel, wallCheckDistance, m_GroundCheckMask,
            QueryTriggerInteraction.Ignore);
        
        var inputDir = new Vector2(strafeInput, forwardInput).normalized;
        Vector3 inputVector = t.right * inputDir.x + t.forward * inputDir.y;
        var isInputDirWall = Physics.Raycast(t.position, inputVector, out var hitInput, wallCheckDistance, m_GroundCheckMask,
            QueryTriggerInteraction.Ignore);

        if (jumpTimer > 0) return;
        
        // Store hit if touching wall
        if (isVelDirWall)
        {
            hit = hitVel;
        }
        else if (isInputDirWall)
        {
            hit = hitInput;
        }
        
        // Cache wall normal
        if (isVelDirWall || isInputDirWall)
        {
            wallNormal = hit.normal;
        }
        else
        {
            // If no wall hit, see if still near wall when a normal is stored, reset if not
            // use to reset wallride state when riding above or below a wall
            if (wallNormal.Equals(Vector3.zero)) return;
            if (!Physics.CheckSphere(transform.position, wallCheckDistance, m_GroundCheckMask,
                    QueryTriggerInteraction.Ignore))
            {
                wallNormal = Vector3.zero;
            }
        }
    }

    private void PerformWallRide(Vector3 normal)
    {
        // is agent air-borne?
        if (IsGrounded)
        {
            CancelWallRide();
            return;
        }
        // Is the normal non-zero/Is there a wall??
        if (wallNormal.Equals(Vector3.zero))
        {
            CancelWallRide();
            return;
        }
        // Is velocity towards/parallel to the wall?
        if (!IsVelocityTowardsWall(normal, wallRideVelocityTowardWallMax))
        {
            CancelWallRide();
            return;
        }
        // Is wall sufficiently vertical?
        if (!IsWallVertical(normal, wallVerticalityMinimum))
        {
            CancelWallRide();
            return;
        }

        // If not wallriding, start wallride
        if (!isWallRide)
        {
            StartWallRide();
        }
        
        wallRideDirection = Vector3.ProjectOnPlane(Velocity, normal);
        Velocity = wallRideDirection;
    }

    private void StartWallRide()
    {
        isWallRide = true;
        isRotateTransform = false;
        m_Gravity = wallrideGravityModifier * defaultGravity;
        SetDrag(wallrideDragModifier*groundDrag);
        wallRideCameraLerpTimer = wallRideCameraLerpDuration;
        CancelWallJumpVelocity();
        
        complexActionPositions[ComplexActionType.WallRide].Add(transform.position);
    }

    // Is the agent moving towards the wall?
    // 1 == moving directly into wall
    // 0 == perpendicular to wall
    // -1 == moving away from wall
    private bool IsVelocityTowardsWall(Vector3 normal, float threshold)
    {
        return Vector3.Dot(Velocity, -normal) >= threshold;
    }

    private bool IsWallVertical(Vector3 normal, float verticalMin)
    {
        return Vector3.Dot(normal, Vector3.up) <= verticalMin;
    }

    private void CancelWallRide()
    {
        if (!isWallRide) return;
        isWallRide = false;
        isRotateTransform = true;
        wallRideDirection = Vector3.zero;
        wallNormal = Vector3.zero;
        m_Gravity = defaultGravity;
        wallRideCameraLerpTimer = wallRideCameraLerpDuration;
    }


    public override void ProcessActions(ActionSegment<float> act, float heuristicMouseScale = 1)
    {
        var forward = Mathf.Clamp(act[forwardIndex], -1.0f, 1.0f);
        var strafe = Mathf.Clamp(act[strafeIndex], -1.0f, 1.0f);
        var yaw = Mathf.Clamp(act[yawIndex], -1.0f, 1.0f)*heuristicMouseScale;
        var pitch = 0f;
        if(usePitch) pitch = Mathf.Clamp(act[pitchIndex], -1.0f, 1.0f)*heuristicMouseScale;
        var jumpInput = act[jumpIndex]>0f;
        var newSprintState = act[sprintIndex]>0;
        var newCrouchState = act[crouchIndex]>0;
        
        ProcessMoveInputs(strafe,forward);
        ProcessSprint(newSprintState);
        ProcessCrouch(newCrouchState);
        
        Move();
        Slide();
        if (useWallRide)
        {
            GetWallNormalForWallRide(strafe, forward);
            PerformWallRide(wallNormal);
        }
        ApplyGravity();
        Jump(jumpInput, jumpForce);
        ApplyWallJumpVelocity();
        Look(yaw, pitch);
    }
    
    /// <summary>
    /// Send movement observations to NavAgent.
    /// 11 observations from JumpMovement + 4 sprint/crouch/wallride observations [15 observations]. 
    /// </summary>
    /// <returns></returns>
    public override List<float> CollectVectorObservations()
    {
        var obs = base.CollectVectorObservations();
        obs.Add(isSprinting?1f:0f);
        obs.Add(isCrouching?1f:0f);
        obs.Add(isSliding?1f:0f);
        obs.Add(isWallRide?1f:0f);
        return obs;
    }

    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawLine(transform.position,transform.position+wallRideDirection);
    }
}
