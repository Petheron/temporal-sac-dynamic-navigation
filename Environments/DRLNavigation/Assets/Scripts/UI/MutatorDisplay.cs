using System;
using TMPro;
using UnityEngine;

namespace UI
{
    public class MutatorDisplay:MonoBehaviour
    {
        
        [SerializeField] private TextMeshProUGUI display;
        [SerializeField] private ModularMovement modularMovement;

        // private Rigidbody rb;

        // private void Awake()
        // {
        //     rb = GetComponent<Rigidbody>();
        // }

        private void FixedUpdate()
        {
            if(!modularMovement) return;
            if(!display.canvas.worldCamera) return;
            if(!display.canvas.worldCamera.enabled) return;
            var currentMutator = modularMovement.currentMutator;
            if (currentMutator == null) return;
            var currentMutatorType = modularMovement.currentMutator.Type;
            
            display.SetText($"Mutator: \n{currentMutatorType}");
        }
    }
}