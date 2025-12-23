using System;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public class PointOctreeTest:MonoBehaviour
{
    [Header("Debug Settings")]
    public bool test;
    public int numPoints = 10;
    public int seed =0;
    public float debugRegionSize = 1f;
    public int minDebugDepth = 3;
    public int maxDebugDepth = 3;
    public bool showAllDepths = false;
    public bool showAllDepthsUpToMax = false;
    public Vector3 debugPoint;
    private List<Vector3> points;

    [Header("Tree Parameters")] 
    public OctreeType treeType = OctreeType.POINT;
    public Vector3 rootPosition;
    public float rootSize = 8f;
    public int maxDepth = 5;
    private SparseOctree tree;
    private OctreeNode[] nodes;
    

    public void Update()
    {
        if (test)
        {
            test = false;
            tree = new SparseOctree(rootPosition, rootSize, maxDepth, treeType);
            if (treeType == OctreeType.POINT)
            {
                points = new List<Vector3>();
                Random.InitState(seed);
                for (int i = 0; i < numPoints; i++)
                {
                    Vector3 p = new Vector3(Random.value, Random.value, Random.value);
                    p *= debugRegionSize;
                    p += rootPosition;
                    points.Add(p);
                    tree.Insert(p,1);
                }
            }
            else
            {
                tree.Construct();
            }
            
            if(showAllDepths) nodes = tree.GetAllNodes();
            else
            {
                int maxShow = showAllDepthsUpToMax ? maxDepth : minDebugDepth;
                nodes = tree.GetNodesAtDepth(minDebugDepth, maxShow);
            }

            var values = tree.RangedSearch(debugPoint, minDebugDepth, maxDebugDepth);
            foreach (var v in values)
            {
                Debug.Log(v);
            }
        }
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.red;
        if (nodes != null)
        {
            foreach (var n in nodes)
            {
                Gizmos.color = n.Data.IsEmpty() ? Color.blue : Color.red;
                Gizmos.DrawWireCube(n.Centre, n.Size*Vector3.one*.99f);
            }
        }
    
        if (points != null)
        {
            foreach (var p in points)
            {
                Gizmos.DrawWireSphere(p,0.05f);
            }
        }
    
        
        
    
    }
}
