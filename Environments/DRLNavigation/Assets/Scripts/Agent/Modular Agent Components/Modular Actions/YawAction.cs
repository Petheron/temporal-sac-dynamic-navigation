using UnityEngine;

public class YawAction:AgentAction
{
    private float rotateSpeed = 1.0f;
    private float minimumRotateThreshold = 1e-4f;
    
    private float yawInput;

    public override void InitializeAgentProperties()
    {
        rotateSpeed = ModularAgentProps.ROTATE_SPEED;
        minimumRotateThreshold = ModularAgentProps.MINIMUM_ROTATE_THRESHOLD;
    }

    protected override void InitializeActionTypes()
    {
        actionTypes.Add(ModularActionType.Yaw);
    }

    public override void ProcessActionInput(float value)
    {
        yawInput = ClampInput(value);
    }

    public override void Perform()
    {
        if (Mathf.Abs(yawInput) < minimumRotateThreshold) return;
        
        float yawDelta = yawInput * rotateSpeed * Time.fixedDeltaTime;

        var rotation = movement.Rotation.eulerAngles;
        movement.SetRotation(Quaternion.Euler(rotation.x, rotation.y + yawDelta, rotation.z));
    }

    public override void Reset()
    {
        yawInput = 0f;
    }
}
