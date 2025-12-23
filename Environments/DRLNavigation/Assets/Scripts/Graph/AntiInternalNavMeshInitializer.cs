
using System;
using Unity.AI.Navigation;
using UnityEngine;

public class AntiInternalNavMeshInitializer:MonoBehaviour
{
    [SerializeField] private bool preventNavMeshOnObject;
    private NavMeshModifierVolume navMeshModifierVolume;
    [SerializeField] private BoxCollider boxCollider;

    private void Awake()
    {
        navMeshModifierVolume = GetComponent<NavMeshModifierVolume>();

        if (!navMeshModifierVolume) return;
        if (!boxCollider) boxCollider = GetComponent<BoxCollider>();
        if (!boxCollider) return;

        Vector3 size = boxCollider.size - 0.01f*Vector3.one;
        Vector3 centre = -0.01f*Vector3.up;
        if (preventNavMeshOnObject)
        {
            size.y = boxCollider.size.y + 0.01f;
            centre.y = 0f;
        }

        navMeshModifierVolume.size = size;
        navMeshModifierVolume.center = centre;
        navMeshModifierVolume.area = 1;
    }
}
