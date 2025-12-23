using System;
using System.Collections.Generic;
using UnityEngine;

public class OccupancyMapObservable : MonoBehaviour
{
    public VisionTag visionTag = VisionTag.Solid;
    public MutatorComponent mutatorComponent;
    protected Vector3 defaultPosition;
    protected Quaternion defaultRotation;
    [SerializeField, ReadOnly] protected Vector3 currentPosition = Vector3.zero;
    [SerializeField, ReadOnly] protected Quaternion currentRotation = Quaternion.identity;

    protected List<Vector3> occupancyNodeOffsets;
    protected List<int[]> occupancyIndices;
    
    protected OccupancyMap occupancyMap;
    protected Rigidbody rb;

    public bool showDebug;
    [SerializeField]protected List<int> debugOccupancyIndicesCount;
    protected List<int> debugOccupancyIndex;

    protected MutatorType ObservableMutatorType => mutatorComponent ? mutatorComponent.mutatorType : MutatorType.None;

    protected virtual void Awake()
    {
        rb = GetComponent<Rigidbody>();

        var t = transform;
        defaultPosition = t.position;
        defaultRotation = t.rotation;
        
        Reset();
    }

    protected virtual void Start()
    {
        occupancyMap = FindObjectOfType<OccupancyMap>();
        if (!occupancyMap) return;
        if (occupancyMap.isMapInitialized)
        {
            InitializeOccupancyNodes();
        }
        else
        {
            occupancyMap.OccupancyMapInitialized += InitializeOccupancyNodes;
        }
    }

    protected virtual void InitializeOccupancyNodes()
    {
        if (occupancyNodeOffsets != null) return;
        occupancyNodeOffsets = new List<Vector3>();
        occupancyIndices = new List<int[]>();
        debugOccupancyIndicesCount = new List<int>();
        debugOccupancyIndex = new List<int>();
        
        // print(occupancyMap.ComputeTagValue(visionTag, ObservableMutatorType));

        Collider[] colliders = gameObject.GetComponentsInChildren<Collider>();
        float cellsize = occupancyMap.CellSize(0);
        foreach (var col in colliders)
        {
            var bounds = col.bounds;
            Vector3 centre = bounds.center;
            Vector3 extents = bounds.extents;
            int x = Mathf.FloorToInt((extents.x)/cellsize);
            int y = Mathf.FloorToInt((extents.y)/cellsize);
            int z = Mathf.FloorToInt((extents.z)/cellsize);
            for (int i = -x; i <= x; i++)
            {
                for (int j = -y; j <= y; j++)
                {
                    for (int k = -z; k <= z; k++)
                    {
                        // only add position if inside collider
                        Vector3 pos = centre + new Vector3(i*cellsize, j*cellsize, k*cellsize);
                        
                        if(Vector3.Distance(col.ClosestPoint(pos), pos) > Mathf.Epsilon) continue;
                        AddNode(pos);
                    }
                }
            }
        }
    }

    private void AddNode(Vector3 position)
    {
        occupancyNodeOffsets.Add(position - transform.position);
        int[] idx = occupancyMap?.GetAllIndicesFromGlobalPosition(position);
        occupancyIndices.Add(idx);
                        
        AddOccupancyIndices(idx, visionTag, ObservableMutatorType);
                        
        debugOccupancyIndicesCount.Add(occupancyMap.GetTotalCountFromGlobalPosition(position,0));
        debugOccupancyIndex.Add(idx[0]);
    }

    /// <summary>
    /// If an occupancy node has moved into a position where it has a different index than before, update the occupancy map accordingly.
    /// This method also changes the old index value as well. 
    /// </summary>
    protected void UpdateOccupancyNodes()
    {
        if (!occupancyMap) return;
        if (occupancyIndices == null) return;
        if (occupancyNodeOffsets == null) return;

        for (int i = 0; i < occupancyIndices.Count; i++)
        {
            int[] oldIdx = occupancyIndices[i];
            int[] newIdx = occupancyMap.GetAllIndicesFromGlobalPosition(currentPosition + occupancyNodeOffsets[i]);
            
            occupancyMap.UpdateDynamicOccupancyIndex(oldIdx, newIdx, visionTag, ObservableMutatorType);
            occupancyIndices[i] = newIdx;

            debugOccupancyIndex[i] = newIdx[0];
            debugOccupancyIndicesCount[i] = occupancyMap.GetTotalCountFromGlobalPosition(currentPosition + occupancyNodeOffsets[i], 0);
        }
    }
    
    protected void AddOccupancyIndices(int[] indices, VisionTag dynamicTag, MutatorType mutatorType)
    {
        occupancyMap.AddNewDynamicOccupancyIndex(indices, dynamicTag, mutatorType);
    }
    
    protected void SubtractOccupancyIndices(int[] indices, VisionTag dynamicTag, MutatorType mutatorType)
    {
        occupancyMap.SubtractDynamicOccupancyIndex(indices, dynamicTag, mutatorType);
    }

    protected void AddAllOccupancyNodes()
    {
        if(occupancyIndices.Count==0) return;
        if (!occupancyMap) return;
        
        foreach (var indices in occupancyIndices)
        {
            AddOccupancyIndices(indices, visionTag, ObservableMutatorType);
        }
    }

    protected void SubtractAllOccupancyNodes()
    {
        if(occupancyIndices.Count==0) return;
        if (!occupancyMap) return;
        
        foreach (var indices in occupancyIndices)
        {
            SubtractOccupancyIndices(indices, visionTag, ObservableMutatorType);
        }
    }

    public virtual void Reset()
    {
        VisibleTags.AssignTag(gameObject, visionTag, ObservableMutatorType);
        transform.SetPositionAndRotation(defaultPosition,defaultRotation);
        if (rb)
        {
            if (!rb.isKinematic)
            {
                rb.velocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }
            rb.ResetInertiaTensor();
        }

        currentPosition = defaultPosition;
        currentRotation = defaultRotation;
        UpdateOccupancyNodes();
    }

    private void OnDisable()
    {
        if (!occupancyMap) return;
        occupancyMap.OccupancyMapInitialized -= InitializeOccupancyNodes;
    }

    protected virtual void OnDrawGizmosSelected()
    {
        if (!showDebug) return;
        if (occupancyNodeOffsets == null) return;
        if (!occupancyMap) return;
        
        Gizmos.color = Color.blue;
        foreach (var p in occupancyNodeOffsets)
        {
            Gizmos.DrawWireCube(p+currentPosition,0.2f*Vector3.one);
        }
    }
}
