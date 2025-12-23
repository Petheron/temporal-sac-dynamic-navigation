using System;
using TMPro;
using UnityEngine;

namespace UI
{
    public class AgentVelocityDisplay:MonoBehaviour
    {
        private enum FrameOfReference
        {
            Local,
            World
        }
        
        [SerializeField] private TextMeshProUGUI display;
        [SerializeField] private FrameOfReference frameOfReference = FrameOfReference.Local;
        [SerializeField] private ModularMovement modularMovement;

        private Rigidbody rb;

        private void Awake()
        {
            rb = GetComponent<Rigidbody>();
        }

        private void FixedUpdate()
        {
            if(!modularMovement) return;
            if(!display.canvas.worldCamera) return;
            if(!display.canvas.worldCamera.enabled) return;

            Vector3 velocity = frameOfReference == FrameOfReference.World ? modularMovement.Velocity: modularMovement.RelativeVelocity;
            
            display.SetText($"Velocity\n{frameOfReference}\nx: {velocity.x:F1}\ny: {velocity.y:F1}\nz: {velocity.z:F1}");
        }
    }
}