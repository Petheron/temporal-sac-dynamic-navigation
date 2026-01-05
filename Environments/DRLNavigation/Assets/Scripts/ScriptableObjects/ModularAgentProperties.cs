using UnityEngine;
using UnityEngine.Serialization;

[CreateAssetMenu(fileName = "ModularAgentProperties", menuName = "Modular Agent Properties", order = 0)]
public class ModularAgentProperties : ScriptableObject
{
    [Header("Model Properties")]
    [FormerlySerializedAs("NUM_ACTION_REPEAT")] 
    public int ACTION_REPETITION = 4;
    public int OBS_STACKS = 4;

    [Header("Observation Properties")] 
    public VectorNormalizationType VECTOR_NORMALIZATION_TYPE = VectorNormalizationType.Componentwise;
    public VectorSpatialOrientationType VECTOR_SPATIAL_ORIENTATION = VectorSpatialOrientationType.WorldSpace;
    public ObservedRotationType OBSERVED_ROTATION_TYPE = ObservedRotationType.YawPitch;

    [Header("Agent Movement Scalars")]
    public float MOVE_ACCELERATION = 32.0f;
    public float MOVE_DRAG = 4.0f;
    public float MOVE_FRICTION = 1.0f;
    public float MOVE_AIR_CONTROL = 0.5f;
    public float MOVE_TURNAROUND_FACTOR = 2f;
    public float GRAVITY_SCALE = 2f;
    public float GRAVITY_FALL_MULTIPLIER = 1.5f;
    public float VELOCITY_LIMIT = 100f;
    
    [Header("Agent Rotation Parameters")]
    public float ROTATE_SPEED = 400.0f;
    public float MINIMUM_ROTATE_THRESHOLD = 1e-4f;

    [Header("Ground Detection Properties")]
    public float GROUNDED_OFFSET = 0.125f;
    public float GROUNDED_RADIUS = 0.25f;
    public LayerMask GROUNDED_MASK;

    [Header("Jump Parameters")] 
    public float JUMP_VELOCITY = 15f;
    public float MINIMUM_JUMP_DURATION = 0.2f;
    public float COYOTE_TIME_DURATION = 0.1f;

    [Header("Dash Parameters")]
    public float DASH_DURATION = 0.25f;
    public float DASH_DISTANCE = 10f;
    public float DASH_COOLDOWN = 3.0f;
    
    [Header("Leap Properties")] 
    public float LEAP_VELOCITY = 40f;
    public float LEAP_ANGLE_MINIMUM = 10f;
    

    [Header("Sprint Parameters")] 
    public bool SPRINT_IS_TOGGLE = true;
    public float SPRINT_MODIFIER = 1.5f;
    public float SPRINT_DURATION_MINIMUM = 0.2f;
    
    [Header("Wallride Parameters")]
    public float WALLRIDE_ACCELERATION_MODIFIER = 0.2f;
    public float WALLRIDE_FALL_SPEED_MODIFIER = 0.025f;
    public float WALLRIDE_GRAVITY_MODIFIER = 2f;
    public float WALLRIDE_DRAG_MODIFIER = 0.1f;
    
    [Header("Walljump Parameters")]
    public float WALLJUMP_VELOCITY = 12.5f;

    [Header("WallClimb Parameters")] 
    public float WALLCLIMB_ACCELERATION = 48f;
    public float WALLCLIMB_SENSITIVITY = 0.8f;
    public float WALLCLIMB_FALL_SPEED_MODIFIER = 0.0001f;
    public float WALLCLIMB_DURATION = 0.5f;
    public float WALL_VERTICALITY_SENSITIVITY = 1.0f;
    public float WALL_CHECK_DISTANCE = 0.6f;

    [Header("Goal Position Properties")] 
    public float MAX_GOAL_DISTANCE = 80f;
    public float NORMALIZATION_DISTANCE = 40f;
    public bool SAMPLE_GOAL_VERTEX_FROM_START_VERTEX = false;

    [Header("Position Generalization Properties")] 
    public bool IS_AGENT_POSITION_RANDOM = true;
    public bool IS_AGENT_ROTATION_RANDOM = true;

    [Header("Reward Scalars")]
    public float GOAL_REWARD = 1f;
    public float FAIL_PENALTY = -1f;
    public float TIMEOUT_PENALTY = 0f;
    public float TIME_STEP_PENALTY = -1.0e-3f;
    
    [Header("Euclidean Reward Properties")]
    public float EUCLIDEAN_TIMEOUT_PENALTY = 0f;
    public float EUCLIDEAN_REWARD_SCALE_FACTOR = 1.0f;
    public bool NORMALIZE_EUCLIDEAN_DISTANCE = true;

    [Header("Termination Conditions")]
    public int DEFAULT_STEP_LIMIT = 5000;
    public float EPSILON_DISTANCE = 1.0f;
    public int EUCLIDEAN_STEP_MAX = 0;
    
}