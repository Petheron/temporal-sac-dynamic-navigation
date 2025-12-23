using System.Collections.Generic;
using UnityEngine;

public enum OctreeType
{
    REGION,
    POINT
}

public class OctreeNode
{
    public bool IsLeaf;
    public OctreeNode Parent = null;
    public OctreeNode[] Children = new OctreeNode[8];
    public VoxelData Data;
    public float Size;
    public Vector3 Centre;
    public int Depth;

    public Voxel GetVoxel()
    {
        return new Voxel(Centre, Size, Data);
    }

}

public class SparseOctree
{
    private OctreeNode root;
    private float size;
    private int maxDepth;
    private Vector3 rootPoint;

    private OctreeType treeType;

    public SparseOctree(Vector3 rootPos, float treeSize, int maximumDepth, OctreeType type)
    {
        root = null;
        rootPoint = rootPos;
        size = treeSize;
        maxDepth = maximumDepth;
        treeType = type;
    }

    public void Construct()
    {
        if (treeType != OctreeType.REGION) return;
        Vector3Int position = Vector3Int.zero;
        RegionInsert(ref root, position,0);
    }

    private void RegionInsert(ref OctreeNode node, Vector3Int position, int depth)
    {
        if (depth > maxDepth) return;
        
        float voxelSize = GetSize(depth);
        Vector3 voxelCentre = GetCubeCentre(position, voxelSize);

        node ??= new OctreeNode();
        
        node.Size = voxelSize;
        node.Centre = voxelCentre;
        node.Depth = depth;

        bool isEmpty = !Physics.CheckBox(voxelCentre, Vector3.one * (0.99f*voxelSize) / 2f);
        node.Data.Value = isEmpty ? 0 : 1;

        if (isEmpty)
        {
            node.IsLeaf = true;
            return;
        }

        if (depth == maxDepth)
        {
            node.IsLeaf = true;
            return;
        }
        
        // recurse through children octants.
        for (int i = 0; i <= 1; i++)
        {
            for (int j = 0; j <= 1; j++)
            {
                for (int k = 0; k <= 1; k++)
                {
                    Vector3Int childPos = new Vector3Int(i, j, k);
                    int idx = GetOctantIndex(childPos);
                    var newPosition = UpdatePosition(position, childPos);
                    RegionInsert(ref node.Children[idx], newPosition, depth+1);
                }
            }
        }
    }

    public void Insert(Vector3 point, int value)
    {
        if (treeType != OctreeType.POINT) return;
        VoxelData data = new VoxelData(value);
        Vector3Int position = Vector3Int.zero;
        InsertImplementation(ref root, point, data, ref position, 0);
    }


    public int[] RangedSearch(Vector3 point, int minDepth, int maxDepth)
    {
        Vector3Int position = Vector3Int.zero;
        List<OctreeNode> nodes = new List<OctreeNode>(){root};
        nodes = RangedSearchImplementation(ref nodes, point, minDepth, maxDepth, ref position, 0);
        List<int> values = new List<int>();
        foreach (var node in nodes)
        {
            int v = node == null ? 0 : node.Data.Value;
            values.Add(v);
        }
        return values.ToArray();
    }
    
    public Voxel Search(Vector3 point, int desiredDepth)
    {
        Vector3Int position = Vector3Int.zero;
        OctreeNode node = SearchImplementation(ref root, point, desiredDepth, ref position, 0);
        return node?.GetVoxel() ?? Voxel.NullVoxel();
    }

    public OctreeNode[] GetAllNodes()
    {
        return GetNodesAtDepth(0,maxDepth);
    }

    public OctreeNode[] GetNodesAtDepth(int minimumDepth, int maximumDepth)
    {
        if (minimumDepth > maximumDepth)
        {
            Debug.LogWarning($"Minimum depth {minimumDepth} should not be greater than maximum depth {maximumDepth}.");
            return null;
        }
        
        List<OctreeNode> nodes = new List<OctreeNode>();
        Queue<OctreeNode> queue = new Queue<OctreeNode>();
        queue.Enqueue(root);
        while (queue.Count>0)
        {
            OctreeNode current = queue.Dequeue();
            if (current.Depth >= minimumDepth) nodes.Add(current);
            else
            {
                if(current.IsLeaf) nodes.Add(current);
            }
            if (current.Depth >= maximumDepth) continue;
            foreach (var child in current.Children)
            {
                if(child!=null) queue.Enqueue(child);
            }
        }
        return nodes.ToArray();
    }
    
    

    private void InsertImplementation(ref OctreeNode node, Vector3 point, VoxelData data, ref Vector3Int position, int depth)
    {
        float voxelSize = GetSize(depth);
        Vector3 voxelCentre = GetCubeCentre(position, voxelSize);

        if (!IsInBounds(point, voxelCentre, voxelSize)) return;
        
        node ??= new OctreeNode();


        node.Data = data;
        node.Size = voxelSize;
        node.Centre = voxelCentre;
        node.Depth = depth;


        if (depth == maxDepth)
        {
            node.IsLeaf = true;
            return;
        }
        
        GetChildOctantPosition(out Vector3Int childPosition, point, voxelCentre);
        // Calculate finer grid index of child to act as parent for next recursion.
        UpdatePosition(ref position, childPosition);
        
        InsertImplementation(ref node.Children[GetOctantIndex(childPosition)], point, data, ref position, ++depth);
    }

    private bool IsInBounds(Vector3 point, Vector3 centre, float voxelSize)
    {
        for (int i = 0; i < 3; i++)
        {
            if (point[i] > centre[i] + voxelSize / 2f || point[i] < centre[i] - voxelSize / 2f) return false;
        }
        return true;

    }

    private Vector3 GetCubeCentre(Vector3Int zeroCornerPosition, float voxelSize)
    {
        return (zeroCornerPosition + 0.5f * Vector3.one) * voxelSize + rootPoint;
    }

    private float GetSize(int depth)
    {
        return  size / Mathf.Pow(2f, depth);
    }

    private void GetChildOctantPosition(out Vector3Int childPos, Vector3 point, Vector3 parentCentre)
    {
        childPos = new Vector3Int();
        for (int i = 0; i < 3; i++)
        {
            childPos[i] = point[i] >= parentCentre[i] ? 1 : 0;
        }
    }

    private int GetOctantIndex(Vector3Int octantPosition)
    {
        return octantPosition.x << 0 | octantPosition.y << 1 | octantPosition.z << 2;
    }

    private void UpdatePosition(ref Vector3Int position, Vector3Int childPosition)
    {
        for (int i = 0; i < 3; i++)
        {
            position[i] = (position[i] << 1) | childPosition[i];
        }
    }
    
    private Vector3Int UpdatePosition(Vector3Int position, Vector3Int childPosition)
    {
        for (int i = 0; i < 3; i++)
        {
            position[i] = (position[i] << 1) | childPosition[i];
        }

        return position;
    }
    

    private OctreeNode SearchImplementation(ref OctreeNode node, Vector3 point, int desiredDepth, ref Vector3Int position, int currentDepth)
    {
        if (node == null)
        {
            return null;
        }

        float s = GetSize(currentDepth);
        Vector3 centre = GetCubeCentre(position, s);

        if (!IsInBounds(point, centre, s)) return null;

        if (node.IsLeaf || currentDepth == desiredDepth)
        {
            return node;
        }
        
        GetChildOctantPosition(out Vector3Int childPosition, point, centre);
        int index = GetOctantIndex(childPosition);
        UpdatePosition(ref position, childPosition);
        
        return SearchImplementation(ref node.Children[index], point, desiredDepth, ref position, ++currentDepth);
    }

    private List<OctreeNode> RangedSearchImplementation(
        ref List<OctreeNode> nodes, Vector3 point, int minSearchDepth,
        int maxSearchDepth, ref Vector3Int position, int currentDepth)
    {
        
        OctreeNode node = nodes[^1];
        if (node == null)
        {
            return nodes;
        }
        
        float s = GetSize(currentDepth);
        Vector3 centre = GetCubeCentre(position, s);
        
        if (!IsInBounds(point, centre, s)) return nodes;

        if (node.IsLeaf || currentDepth == maxDepth)
        {
            while (nodes.Count < (maxSearchDepth - minSearchDepth) + 1)
            {
                nodes.Add(node);
            }
            return nodes;
        }

        if (currentDepth >= minSearchDepth && currentDepth<maxSearchDepth)
        {
            nodes.Add(node);
        }
        
        GetChildOctantPosition(out Vector3Int childPosition, point, centre);
        int index = GetOctantIndex(childPosition);
        UpdatePosition(ref position, childPosition);

        nodes[^1] = node.Children[index];

        return RangedSearchImplementation(ref nodes, point, minSearchDepth, maxSearchDepth, ref position,
            ++currentDepth);

    }
}
