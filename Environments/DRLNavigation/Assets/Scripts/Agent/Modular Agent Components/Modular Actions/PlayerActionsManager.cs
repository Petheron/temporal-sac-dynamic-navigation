using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerActionsManager:MonoBehaviour
{
    private List<ModularActionType> actionTypes;

    [SerializeField,ReadOnly] private List<InputAction> inputActions;

    private ModularPlayerActions playerActions;

    [SerializeField,ReadOnly] private float[] actionArray;

    public void Initialize(List<ModularActionType> newActionTypes)
    {
        inputActions = new List<InputAction>();
        playerActions = new ModularPlayerActions();
        playerActions.Enable();

        actionTypes = newActionTypes;

        foreach (var actionType in actionTypes)
        {
            var inputAction = playerActions.FindAction(actionType.ToString());
            if (inputAction == null) return;
            inputActions.Add(inputAction);
        }
        actionArray = new float[inputActions.Count];
    }

    private void Update()
    {
        ReadValues();
    }

    private void ReadValues()
    {
        for (int i = 0; i < inputActions.Count; i++)
        {
            var action = inputActions[i];
            if (action.type == InputActionType.Button)
            {
                if(action.IsPressed()) actionArray[i] = 1.0f;
            }
            else
            {
                actionArray[i] = action.ReadValue<float>();
            }
        }
    }


    public void HeuristicInputsToActions(in ActionSegment<float> continuousActionsOut)
    {
        for (int i = 0; i < continuousActionsOut.Length; i++)
        {
            if (i >= actionArray.Length) break;
            continuousActionsOut[i] = actionArray[i];
        }
        actionArray = new float[inputActions.Count];
    }
}
