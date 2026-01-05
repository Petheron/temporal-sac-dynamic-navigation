using UnityEngine;
using UnityEngine.Serialization;

[CreateAssetMenu(fileName = "MonolithicAgentProperties", menuName = "Monolithic Agent Properties", order = 0)]
public class MonolithicAgentProperties : ScriptableObject
{
    [Header("Model Properties")]
    [FormerlySerializedAs("NUM_ACTION_REPEAT")] 
    public int ACTION_REPETITION = 4;
    public int OBS_STACKS = 4;

    [Header("Agent Movement Scalars")]
    public float GRAVITY_SCALE = 2f;
    public float VELOCITY_LIMIT = 100f;
    
    [Header("Agent Rotation Parameters")]
    public float ROTATE_SPEED = 400.0f;
    
    [Header("Movement Modifiers")]
    public float MOVE_SPEED = 8.0f;
    public float MOVE_FORCE = 60.0f;
    public float GROUND_DRAG = 5.0f;
    public float AIR_DRAG = 1.0f;
    public float AIR_SPEED_MODIFIER = 0.5f;
    public float AIR_SPEED_DECAY_MODIFIER = 0.5f;
    

    [Header("Ground Detection Properties")]
    public float GROUNDED_OFFSET = 0.125f;
    public float GROUNDED_RADIUS = 0.25f;
    public LayerMask GROUNDED_MASK;

    [Header("Jump Parameters")] 
    public float MINIMUM_JUMP_DURATION = 0.2f;
    public float JUMP_FORCE = 10.0f;
    public float JUMP_HEIGHT = 2.2f;
    

    [Header("Sprint Parameters")] 
    public float SPRINT_MODIFIER = 1.5f;

    [Header("Crouch Parameters")] 
    public float CROUCH_SPEED_MODIFIER = 0.25f;
    public float CROUCH_LERP_DURATION = 0.1f;
    public float CROUCH_HEIGHT_OFFSET = 0.5f;
    
    [Header("Slide Parameters")]
    public float SLIDE_FORCE = 10f;
    public float SLIDE_SPEED = 20f;
    public float SLIDE_DURATION = 0.5f;
    
    [Header("Wallride Parameters")]
    public float WALLRIDE_GRAVITY_MODIFIER = 2f;
    public float WALLRIDE_DRAG_MODIFIER = 0.1f;
    
    [Header("Walljump Parameters")]
    public float WALLJUMP_VELOCITY = 12.5f;
    public float WALLRIDE_JUMP_FORCE = 10f;
    public float WALLRIDE_JUMP_FORCE_MODIFIER = 0.66f;
    public float WALLJUMP_DURATION = 0.25f;
    
    [Space(10)]
    public float WALLRIDE_CHECK_DISTANCE = 1.0f;
    public float WALLRIDE_VERTICALITY_MINIMUM = 0.1f;
    public float WALLRIDE_VELOCITY_TOWARDS_WALL_MAX = 0.1f;
    [Space(10)]
    public float WALLRIDE_DUTCH_ANGLE = 15f;
    public float WALLRIDE_CAMERA_LERP_DURATION = 0.1f;

    [Header("Goal Position Properties")] 
    public float MAX_GOAL_DISTANCE = 80f;
    public float NORMALIZATION_DISTANCE = 40f;
    public bool SAMPLE_GOAL_VERTEX_FROM_START_VERTEX = false;

    [Header("Environment Toggles")] 
    public bool IS_AGENT_POSITION_RANDOM = true;
    public bool IS_AGENT_ROTATION_RANDOM = true;

    [Header("Reward Scalars")]
    public float GOAL_REWARD = 1f;
    public float FAIL_PENALTY = -1f;
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