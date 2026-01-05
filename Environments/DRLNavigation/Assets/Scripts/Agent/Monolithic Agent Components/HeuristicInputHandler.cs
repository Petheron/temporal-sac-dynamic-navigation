using Unity.MLAgents.Actuators;
using UnityEngine;


public class HeuristicInputHandler:MonoBehaviour
{
    private float forwardAction;
    private float strafeAction;
    private float lookXAction;
    private float lookYAction;
    private float jumpAction;
    private float dashAction;
    private float sprintAction;
    private float crouchAction;

    private float[] actionArray;

    public float maximumMouseDelta = 5.0f;

    private ModularAgentProperties m_ModularAgentProperties;

    int forwardIndex = 0;
    int strafeIndex = 1;
    int yawIndex = 2;
    int pitchIndex = 3;
    int jumpIndex = 4;
    int sprintIndex = 5;
    int crouchIndex = 6;
    
    private void Awake()
    {
        actionArray = new float[10];
    }

    public void UpdateActionIndices(bool usePitch)
    {
        if (usePitch) return;
        pitchIndex = -1;
        jumpIndex -= 1;
        sprintIndex -= 1;
        crouchIndex -= 1;
    }

    private void Update()
    {
        SetInputs();
    }

    private void SetInputs()
    {
        forwardAction = Mathf.Clamp(Input.GetAxis("Vertical"), -1.0f, 1.0f);
        strafeAction = Mathf.Clamp(Input.GetAxis("Horizontal"), -1.0f, 1.0f);
        lookXAction = Mathf.Clamp(Input.GetAxis("Look X"), -1f, 1f);
        lookYAction = Mathf.Clamp(Input.GetAxis("Look Y"), -1f, 1f);
        if(jumpAction<=0) jumpAction = Input.GetButtonDown("Jump") ? 1f : 0f;
        if(dashAction<=0) dashAction = Input.GetButtonDown("Dash") ? 1f : 0f;
        sprintAction = Input.GetButton("Sprint") ? 1f : 0f;
        crouchAction = Input.GetButton("Crouch") ? 1f : 0f;

        actionArray[forwardIndex] = forwardAction;
        actionArray[strafeIndex] = strafeAction;
        actionArray[yawIndex] = lookXAction;
        if(pitchIndex!=-1) actionArray[pitchIndex] = lookYAction;
        actionArray[jumpIndex] = jumpAction;
        actionArray[sprintIndex] = sprintAction;
        actionArray[crouchIndex] = crouchAction;
    }

    
    public void HeuristicInputsToActions(in ActionSegment<float> continuousActionsOut)
    {
        for (int i = 0; i < continuousActionsOut.Length; i++)
        {
            continuousActionsOut[i] = actionArray[i];
        }
        
        ResetBooleanActions();
    }

    /// <summary>
    /// Resets boolean actions to their negative state to prevent successive uses during heuristic testing.
    /// </summary>
    private void ResetBooleanActions()
    {
        jumpAction = 0f;
        dashAction = 0f;
        sprintAction = 0f;
        crouchAction = 0f;
    }
}
