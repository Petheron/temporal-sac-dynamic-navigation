using System;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.AI;
using Random = UnityEngine.Random;

public enum PointSamplingType
{
    Manual=0,
    Uniform=1,
    PoissonDisk=2,
}

[CreateAssetMenu(fileName = "VertexSamplingManager", menuName = "VertexSamplingManager", order = 0)]
public class VertexSamplingManager : ScriptableObject
{
    [Header("General Sampling Parameters")]
    [SerializeField] private PointSamplingType samplingType = PointSamplingType.Uniform;
    [SerializeField] private bool alwaysIncludeManualNodes = true;
    private List<Vector3> manualPositions;
    [SerializeField] private Vector2Int gridDim = new Vector2Int(5,5);
    [SerializeField] private float samplingHalfHeight = 20f;
    [SerializeField] private float navMeshValidDist = 0.1f;
    [SerializeField] private int seed = 0;
    [SerializeField] private LayerMask raycastLayerMask;
    
    [Header("Edge Connection Parameters")]
    [SerializeField] private float minDist = 2.0f;
    [SerializeField] private float maxDist = 2.0f;
    [SerializeField] private int kNeighbours = 0;
    
    [Header("Uniform Sampling Parameters")]
    [SerializeField] private float cubeSize = 1.0f;
    
    [Header("Poisson Disk Sampling Parameters")] 
    [SerializeField] private float poissonMinRadius = 1.0f;
    [SerializeField] private int poissonSamplingK = 30;
    [SerializeField] private float poissonRadiusFactor = 2.0f;
    [SerializeField] private bool poissonUseRandomInitialPos = true;
    private Digraph digraph;
    
    private EnvironmentParameters m_EnvironmentParameters;

    public event Action OnGraphMade;
    public event Action OnBeginEvaluationTrajectoryInitialization;

    
    public static void IncrementEdgeIndex(ref int startIndex, ref int edgeIndex, int numEdges)
    {
        edgeIndex++;
        if (edgeIndex >= numEdges)
        {
            edgeIndex = 0;
            startIndex++;
        }
    }

    public static void IncrementVertexIndices(ref int startIndex, ref int goalIndex, int numVertices)
    {
        // DO nothing if complete
        if (startIndex == numVertices) return;
        // increment goal index. Double increment if start and goal same.
        goalIndex++;
        if (goalIndex == startIndex) goalIndex++;
        // if goal incremented past num vertices, then increment start index
        if (goalIndex >= numVertices)
        {
            startIndex++;
            goalIndex = 0;
        }
    }
    
    public void CreateGraph()
    {
        ClearGraph();
        if (!digraph)
        {
            var g = new GameObject("Digraph");
            digraph = g.AddComponent<Digraph>();
        }
        if (!digraph) return;
        SetPropertiesFromEnvironmentParams();
        digraph.SetMinMaxDistances(minDist,maxDist);
        // ensure odd grid
        if ((gridDim.x * gridDim.y) % 2 == 0)
        {
            for (int i = 0; i < 2; i++)
            {
                gridDim[i] += (gridDim[i] + 1) % 2;
            }
        }
        
        // Cache Random state to reload after digraph node placement
        var oldState = Random.state;
        Random.InitState(seed);
        
        GeneratePositions(gridDim, samplingHalfHeight, out List<Vector3> positions, out List<Transform> parents);
        digraph.AddVertices(positions,parents);
        digraph.GenerateEdges();
        digraph.PruneEdges(kNeighbours);
        Random.state = oldState;
        
        OnBeginEvaluationTrajectoryInitialization?.Invoke();
        
        // Debug.LogWarning($"Digraph complete: {digraph!=null} with {digraph.NumVertices()} positions.");
    }

    public void InvokeOnGraphMade()
    {
        OnGraphMade?.Invoke();
    }

    public void SetVertexSaveFileName(String fileName)
    {
        digraph.SetVertexFileName(fileName);
    }

    private void SetPropertiesFromEnvironmentParams()
    {
        m_EnvironmentParameters ??= Academy.Instance.EnvironmentParameters;
        if (m_EnvironmentParameters == null) return;

        samplingType = (PointSamplingType) (int) m_EnvironmentParameters.GetWithDefault("sampling_type", (int)samplingType);
        samplingHalfHeight = m_EnvironmentParameters.GetWithDefault("sampling_half_height", samplingHalfHeight);
        poissonMinRadius = m_EnvironmentParameters.GetWithDefault("poisson_min_radius", poissonMinRadius);
    }

    public void ClearGraph()
    {
        if (!digraph) return;
        digraph.RemoveVertices();
        DestroyImmediate(digraph.gameObject);
        digraph = null;
    }

    public void UpdateEdges()
    {
        if (!digraph) return;
        digraph.SetMinMaxDistances(minDist,maxDist);
        digraph.UpdateEdges();
    }

    public void PruneEdges()
    {
        if (!digraph) return;
        UpdateEdges();
        digraph.PruneEdges(kNeighbours);
    }

    private void GeneratePositions(Vector2Int dim, float halfHeight, out List<Vector3> positions, out List<Transform> parents)
    {
        positions = new List<Vector3>();
        parents = new List<Transform>();

        if (!digraph) return;

        if (alwaysIncludeManualNodes)
        {
            ManualSampling(ref positions, ref parents);
        }

        switch (samplingType)
        {
            case PointSamplingType.Manual:
                if (alwaysIncludeManualNodes) break;
                ManualSampling(ref positions, ref parents);
                break;
            case PointSamplingType.Uniform:
                UniformSampling(dim, halfHeight, cubeSize, ref positions, ref parents);
                break;
            case PointSamplingType.PoissonDisk:
                PoissonSampling(dim,halfHeight,poissonMinRadius, poissonSamplingK, poissonRadiusFactor, poissonUseRandomInitialPos, ref positions, ref parents);
                break;
            default:
                throw new ArgumentOutOfRangeException();
        }
    }

    private void ManualSampling(ref List<Vector3> positions, ref List<Transform> parents)
    {
        FindManualVertexPositions(out var manualPos);
        foreach (var p in manualPos)
        {
            GetValidNavMeshPositions(p, navMeshValidDist, ref positions, ref parents);
        }
        
    }

    private void FindManualVertexPositions(out List<Vector3> manualPos)
    {
        var manualVerts = FindObjectsOfType<Vertex>();
        manualPos = manualVerts.Select(v => v.GetPosition()).ToList();
    }

    private void UniformSampling(Vector2Int dim, float halfHeight, float cellSize, ref List<Vector3> positions, ref List<Transform> parents)
    {
        if (!digraph) return;
        
        Vector2Int halfDim = dim / 2;
        Vector2 halfGrid = (Vector2)halfDim * cellSize;
        Vector3 bottomLeft = new Vector3(-halfGrid.x, 0, -halfGrid.y);

        for (int i = 0; i < dim.x; i++)
        {
            for (int j = 0; j < dim.y; j++)
            {
                Vector3 samplePos = bottomLeft + new Vector3(i,0,j) * cellSize;
                GetValidNavMeshPositions(samplePos, halfHeight, ref positions, ref parents);
            }
        }
    }
    
    private void PoissonSampling(Vector2Int dim, float halfHeight, float minRadius, int samplingK, float rFactor, bool useRandomSeed, ref List<Vector3> positions, ref List<Transform> parents)
    {
        if (!digraph) return;

        List<Vector3> poissonPositions = new List<Vector3>();
        List<Vector3> activePoints = new List<Vector3>();

        int N = 2;
        int[,] grid = new int[dim.x, dim.y];

        for (int i = 0; i < dim.x; i++)
        {
            for (int j = 0; j < dim.y; j++)
            {
                grid[i, j] = -1;
            }
        }
        float cellSize = minRadius/Mathf.Sqrt(N);
        Vector2 gridSize = cellSize * (Vector2) dim;

        if (positions.Count>0)
        {
            foreach (var p in positions)
            {
                var pos = p;
                pos.y = 0f;
                if (IsValidPoissonPos(ref grid, ref poissonPositions, gridSize, pos, minRadius, cellSize))
                {
                    poissonPositions.Add(pos);
                    InsertPosition(ref grid, pos, poissonPositions.Count-1, gridSize, cellSize);
                    activePoints.Add(pos);
                }
            }
            positions.Clear();
            parents.Clear();
        }

        Vector3 pos0 = useRandomSeed?
            new Vector3(
                Random.Range(-gridSize.x/2, gridSize.x/2),
                0f,
                Random.Range(-gridSize.y/2, gridSize.y/2)
                )
            :Vector3.zero;
        if (IsValidPoissonPos(ref grid, ref poissonPositions, gridSize, pos0, minRadius, cellSize))
        {
            poissonPositions.Add(pos0);
            InsertPosition(ref grid, pos0,poissonPositions.Count-1, gridSize, cellSize);
            activePoints.Add(pos0);
        }

        while (activePoints.Count>0)
        {
            if (activePoints.Count == 0) break;
            int i = Random.Range(0, activePoints.Count);
            // int i = randomState.Next(0, activePoints.Count);
            Vector3 activePoint = activePoints[i];
            bool foundValidPos = false;

            for (int k = 0; k < samplingK; k++)
            {
                // theta = azimuthal rot (xz-plane) [0,2*PI]
                float theta = Random.Range(-Mathf.PI, Mathf.PI);
                // float theta = ((float)randomState.NextDouble() * 2f - 1.0f) * Mathf.PI;
                float phi = Mathf.PI/2f;
                float r = Random.Range(minRadius, rFactor * minRadius);
                // float r = (float)randomState.NextDouble() * minRadius*(rFactor-1f) + minRadius;
                r += Mathf.Epsilon;

                float x = r * Mathf.Sin(phi) * Mathf.Cos(theta);
                float y = 0f;
                float z = r * Mathf.Sin(phi) * Mathf.Sin(theta);

                Vector3 newPos = activePoint + new Vector3(x, y, z);

                foundValidPos = IsValidPoissonPos(ref grid, ref poissonPositions, gridSize, newPos, minRadius, cellSize);
                // Is valid position?
                // If so break, add to active list and positions list
                if (!foundValidPos) continue;
                poissonPositions.Add(newPos);
                InsertPosition(ref grid, newPos, poissonPositions.Count-1, gridSize, cellSize);
                activePoints.Add(newPos);
                break;
            }
            // if no valid pos found around active point, remove it
            if (!foundValidPos) activePoints.RemoveAt(i);
        }
        // loop through positions, check if they are on NavMesh and obtain parent object below if on NavMesh, and add to parents.
        // otherwise remove from positions and empty grid cell.
        foreach (var p in poissonPositions)
        {
            GetValidNavMeshPositions(p,halfHeight,ref positions, ref parents);
        }
    }
    
    /// <summary>
    /// Insert a position into the Poisson Grid. Assumed that position is within gridSize.
    /// Places current index in poissonPosition into the grid cell. -1 value if cell is empty.
    /// </summary>
    private void InsertPosition(ref int[,] grid, Vector3 pos, int index, Vector2 gridSize, float cellSize)
    {
        Vector2 gridOffset = gridSize / cellSize - Vector2.one;
        gridOffset /= 2f;

        int x = Mathf.RoundToInt(pos.x / cellSize);
        int z = Mathf.RoundToInt(pos.z / cellSize);
        x += (int)gridOffset.x;
        z += (int)gridOffset.y;
        
        grid[x, z] = index;
    }

    /// <summary>
    /// Check if a position can be placed into the poisson grid.
    /// </summary>
    /// <returns>Was the position valid?</returns>
    private bool IsValidPoissonPos(ref int[,] grid, ref List<Vector3> positions, Vector2 gridSize, Vector3 pos, float r, float cellSize)
    {
        //Return false if position not within grid
        Vector2 halfGrid = gridSize / 2f;
        if(Mathf.Abs(pos.x) > halfGrid.x || Mathf.Abs(pos.z) > halfGrid.y)
        {
            return false;
        }
        
        // Index of considered position. Apply grid offset to ensure indices are correct.
        Vector2 gridOffset = gridSize / cellSize - Vector2.one;
        gridOffset /= 2f;
        
        int x = Mathf.RoundToInt(pos.x / cellSize) + (int)gridOffset.x;
        int z = Mathf.RoundToInt(pos.z / cellSize) + (int)gridOffset.y;
        
        // prevent leaving the grid when checking neighbours
        int x0 = Mathf.Max(x-1, 0);
        int x1 = Mathf.Min(x+1, 2*(int)gridOffset.x);
        int z0 = Mathf.Max(z-1, 0);
        int z1 = Mathf.Min(z+1, 2*(int)gridOffset.y);

        // Check neighbours of position.
        for (int i = x0; i <= x1; i++)
        {
            for (int k = z0; k <= z1; k++)
            {
                // if neighbour empty, ignore
                if(grid[i,k]==-1) continue;
                Vector3 testPos = positions[grid[i, k]];
                // Return false if any neighbouring position if closer than r.
                // Also corrects for if considered cell is occupied.
                if (Vector3.Distance(pos, testPos) < r) return false;
            }
        }
        return true;
    }

    private void GetValidNavMeshPositions(Vector3 samplePosition, float halfHeight, ref List<Vector3> positions, ref List<Transform> parents)
    {
        RaycastHit[] hits = Physics.RaycastAll(samplePosition + halfHeight * Vector3.up, Vector3.down, 2 * halfHeight,
            raycastLayerMask, QueryTriggerInteraction.Ignore);
        if (hits.Length <= 0) return;

        foreach (var hit in hits)
        {
            Vector3 pos = hit.point;
            Transform parent = hit.transform;
            if(!NavMesh.SamplePosition(pos, out NavMeshHit navHit, navMeshValidDist, raycastLayerMask))
            {
                continue;
            }
            pos = navHit.position;
            positions.Add(pos);
            parents.Add(parent);
        }
    }

    public int NumVertices()
    {
        return digraph.NumVertices();
    }

    /// <summary>
    /// Get a vertex for the provided index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public Vertex GetVertex(int index)
    {
        return digraph.GetVertexAt(index);
    }

    public int GetVertexIndex(Vertex v)
    {
        return digraph.FindVertexIndex(v);
    }

    // /// <summary>
    // /// Sample random vertex from the digraph.
    // /// </summary>
    // /// <returns></returns>
    // public Vertex GetRandomVertex()
    // {
    //     return digraph.GetRandomVertex();
    // }

    /// <summary>
    /// Select a random edge vertex for the considered vertex.
    /// </summary>
    /// <param name="fromVertex"></param>
    /// <returns></returns>
    public Vertex SampleEdgeVertex(Vertex fromVertex)
    {
        return fromVertex.SampleEdge();
    }

    /// <summary>
    /// Sample a random vertex from the digraph within SampleRadius of the world origin and a minimum distance away from a position to avoid.
    /// </summary>
    /// <param name="sampleRadius"></param>
    /// <param name="positionToAvoid"></param>
    /// <param name="minDistanceBetweenPoints"></param>
    /// <returns></returns>
    public Vertex SampleVertex(float sampleRadius, Vector3 positionToAvoid, float minDistanceBetweenPoints=0f)
    {
        Debug.Assert(sampleRadius >= 0);

        Vertex vertex = null;

        for (int i = 0; i < 1000; i++)
        {
            vertex = digraph.GetRandomVertex();
            if(!vertex) continue;
            Vector3 samplePosition = vertex.GetPosition();
            // Is sample within sample radius?
            float sqrDist = (samplePosition - Vector3.zero).magnitude;
            if (sqrDist > sampleRadius) continue;
            // Is sample too close to position to avoid?
            float avoidSqrDist = (positionToAvoid-samplePosition).magnitude;
            if (avoidSqrDist <= minDistanceBetweenPoints) continue;
            // Return valid vertex position
            return vertex;
        }
        // No valid vertex found
        return vertex;
    }

    public void UpdateEdgeWeight(Vertex startVertex, Vertex goalVertex, float newWeight)
    {
        startVertex.UpdateEdgeWeight(goalVertex,newWeight);
    }
}
