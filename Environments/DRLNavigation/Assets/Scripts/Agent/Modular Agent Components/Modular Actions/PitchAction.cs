using System.Collections.Generic;
using Cinemachine.Utility;
using UnityEngine;

public class PitchAction:AgentAction
{
    private float rotateSpeed = 1.0f;
    private float minimumRotateThreshold = 1e-4f;
    
    private float pitchInput;

    public override void InitializeAgentProperties()
    {
        rotateSpeed = ModularAgentProps.ROTATE_SPEED;
        minimumRotateThreshold = ModularAgentProps.MINIMUM_ROTATE_THRESHOLD;
    }

    protected override void InitializeActionTypes()
    {
        actionTypes.Add(ModularActionType.Pitch);
    }

    public override void ProcessActionInput(float value)
    {
        pitchInput = ClampInput(value);
        
    }

    public override void Perform()
    {
        if (Mathf.Abs(pitchInput) < minimumRotateThreshold) return;
        float pitchDelta = pitchInput * rotateSpeed*Time.fixedDeltaTime;
        var rotation = movement.Rotation.eulerAngles;
        movement.SetRotation(Quaternion.Euler(ClampPitch(rotation.x - pitchDelta), rotation.y, rotation.z));
    }

    private float ClampPitch(float newPitch)
    {
        if (newPitch > 180) newPitch -= 360f;
        return Mathf.Clamp(newPitch, -90f, 90f);
    }

    public override void Reset()
    {
        pitchInput = 0f;
    }
}