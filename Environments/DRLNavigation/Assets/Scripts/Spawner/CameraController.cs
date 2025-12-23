using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Cinemachine;
using Unity.MLAgents;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    [ReadOnly, SerializeField] private List<Camera> cameras;
    [ReadOnly, SerializeField] private int currentCameraIndex;
    [ReadOnly, SerializeField] private Camera currentCamera;

    private LevelSpawner levelSpawner;

    public Camera CurrentCamera => currentCamera;

    private void Awake()
    {
        cameras = new List<Camera>();
        levelSpawner = GetComponent<LevelSpawner>();
    }

    private void Update()
    {
        if (Input.GetButtonDown("CameraCycle") && cameras.Count>0)
        {
            CycleToNewCamera((int)Mathf.Sign(Input.GetAxis("CameraCycle")));
        }
    }

    public void AddCamera(Camera newCamera)
    {
        if(!newCamera) return;
        if (cameras.Contains(newCamera)) return;
        DisableCamera(newCamera);
        cameras.Add(newCamera);
        if (cameras.Count == 1)
        {
            currentCameraIndex = 0;
        }
        CycleToNewCamera(0);
    }

    private void CycleToNewCamera(int cycleDirection)
    {
        DisableCamera(currentCamera);
        currentCameraIndex += Math.Clamp(cycleDirection, -1, 1);
        currentCameraIndex %= cameras.Count;
        if (currentCameraIndex < 0) currentCameraIndex += cameras.Count;
        SetCurrentCamera();
    }

    private void SetCurrentCamera()
    {
        if(currentCameraIndex<0) return;
        if (currentCameraIndex >= cameras.Count) return;
        currentCamera = cameras[currentCameraIndex];
        EnableCamera(currentCamera);
    }

    private void EnableCamera(Camera cam)
    {
        if (!cam) return;
        
        var virtCam = cam.gameObject.GetComponent<CinemachineVirtualCamera>();
        if (virtCam)
        {
            virtCam.enabled = true;
            virtCam.Priority = 10;
        }
        else
        {
            cam.enabled = true;
        }
    }

    private void DisableCamera(Camera cam)
    {
        if (!cam) return;
        // cam.enabled = false;
        var virtCam = cam.gameObject.GetComponent<CinemachineVirtualCamera>();
        if (virtCam)
        {
            virtCam.enabled = false;
            virtCam.Priority = 0;
        }
        else
        {
            cam.enabled = false;
        }
    }

    private void DisableAllCameras()
    {
        if (cameras == null) return;
        foreach (var cam in cameras.Where(cam => cam))
        {
            DisableCamera(cam);
        }
    }

    public void RemoveAllCameras()
    {
        cameras.Clear();
    }

    public void RemoveCamera(Camera cameraToRemove)
    {
        if(!cameraToRemove) return;
        if (!cameras.Contains(cameraToRemove)) return;
        cameras.Remove(cameraToRemove);
        int cycleDir = 0;
        if (cameraToRemove != currentCamera)
        {
            for (int i = 0; i < cameras.Count; i++)
            {
                if (currentCamera == cameras[i])
                {
                    cycleDir = currentCameraIndex - i;
                    break;
                }
            }
        }
        CycleToNewCamera(cycleDir);
    }

    public void ClearCameras()
    {
        cameras.Clear();
    }
    
    
}
