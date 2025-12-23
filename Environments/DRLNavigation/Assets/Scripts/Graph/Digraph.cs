using System;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;
using Vector3 = UnityEngine.Vector3;
using SystemRandom = System.Random;

public class Digraph: MonoBehaviour
{
    public float minimumDistance = 2f;
    public float maximumDistance = 5;
    private const float InitialWeight = 0f;
    private List<Vertex> vertices;

    private String vertexSaveName = "Vertices";
    public String vertexLoadName = "Vertices";
    
    [Header("Test Variables")]
    public Vector3 start;
    public Vector3 goal;
    public List<Vertex> shortestPath;
    
    public bool showGizmos = true;

    public Digraph()
    {
        vertices = new List<Vertex>();
        shortestPath = new List<Vertex>();
    }

    public void SetVertexFileName(String fileName)
    {
        vertexSaveName = fileName;
        vertexLoadName = fileName;
    }

    public int FindVertexIndex(Vertex v)
    {
        return vertices.IndexOf(v);
    }
    

    public void SetMinMaxDistances(float min, float max)
    {
        minimumDistance = min;
        maximumDistance = max;

        foreach (var v in vertices)
        {
            v.SetDistanceConditions(minimumDistance, maximumDistance);
        }
    }

    public void AddVertices(List<Vector3> positions, List<Transform> parents)
    {
        for (int i = 0; i < positions.Count; i++)
        {
            AddVertex($"Vertex{i}", positions[i], parents[i]);
        }
    }

    public void RemoveVertices()
    {
        for (int i = vertices.Count-1; i >= 0; i--)
        {
            RemoveVertex(vertices[i]);
        }
    }

    public Vertex GetVertexAt(int index)
    {
        if (index < 0 || index >= vertices.Count) return null;
        return vertices[index];
    }

    public Vertex GetRandomVertex()
    {
        int idx = Random.Range(0, NumVertices());
        // int idx = randomState.Next(0,NumVertices());
        Vertex v = GetVertexAt(idx);
        return v;
    }

    private void AddVertex(string name, Vector3 pos, Transform parent)
    {
        var go = new GameObject(name)
        {
            transform = {parent = parent}
        };
        Vertex v = go.AddComponent<Vertex>();
        v.SetPosition(pos);
        v.SetDistanceConditions(minimumDistance, maximumDistance);
        v.SetInitialWeight(InitialWeight);
        v.SetDynamic();
        vertices.Add(v);
    }

    public void RemoveVertex(Vertex v)
    {
        if (!vertices.Contains(v)) return;
        vertices.Remove(v);
        foreach (var vert in vertices)
        {
            vert.RemoveEdge(v);
        }

        if (!v) return;
        var go = v.gameObject;
        DestroyImmediate(go);
    }

    public int NumVertices()
    {
        return vertices.Count;
    }

    public int NumEdges()
    {
        int n = 0;
        foreach (var node in vertices)
        {
            n += node.NumEdges();
        }

        return n;
    }

    public void AddEdge(Vertex fromVertex, Vertex toVertex)
    {
        fromVertex.AddEdge(toVertex);
    }

    public void GenerateEdges()
    {
        foreach (var u in vertices)
        {
            foreach (var v in vertices)
            {
                AddEdge(u,v);
            }
        }
    }

    public void UpdateEdges()
    {
        foreach (var u in vertices)
        {
            foreach (var v in vertices)
            {
                u.UpdateEdgeValidity(v);
            }
        }
    }

    public void PruneEdges(int k)
    {
        foreach (var v in vertices)
        {
            v.PruneEdges(k);
        }
    }

    public List<Vertex> GetShortestPath(Vector3 start, Vector3 goal)
    {
        Vertex startVertex = GetNearestVertex(start);
        Vertex goalVertex = GetNearestVertex(goal);

        return GetShortestPath(startVertex, goalVertex);
    }

    /// <summary>
    /// Finds shortest path between supplied start and end nodes.
    /// Utilizes Djikestra's algorithm to determine node distances.
    /// </summary>
    /// <returns>Shortest path of vertices</returns>
    public List<Vertex> GetShortestPath(Vertex startVertex, Vertex goalVertex)
    {
        List<Vertex> path = new List<Vertex>();
        foreach (var v in vertices)
        {
            v.SetDistance(v.Equals(startVertex)? 0f: Mathf.Infinity);
        }

        List<Vertex> unvisited = new List<Vertex>();
        unvisited.AddRange(vertices);
        Vertex current = startVertex;
        

        while (unvisited.Count>0)
        {
            // Update distances for all edges to current node
            for (int i = 0; i < current.NumEdges(); i++)
            {
                Vertex e = current.GetEdges()[i];
                if (!unvisited.Contains(current.GetEdges()[i])) continue;
                float dist = Mathf.Min(e.GetDistance(), current.GetDistance() + current.SampleEdgeWeight(e));
                e.SetDistance(dist);
            }
            
            // consider current visited and add to the path
            unvisited.Remove(current);

            // new current is one with shortest distance (greedy selection)
            float minDist = Mathf.Infinity;
            Vertex newCurrent = null;
            foreach (var u in unvisited)
            {
                if (u.GetDistance() < minDist)
                {
                    minDist = u.GetDistance();
                    newCurrent = u;
                }
            }
            current = newCurrent;
        }
        
        // Construct the path. Break ties with a list if necessary.
        Vertex currentVertex = startVertex;
        path.Add(currentVertex);
        while (!currentVertex.Equals(goalVertex))
        {
            // Debug.Log($"c: {c}, dist {c.distance}");
            float minDistance = Mathf.Infinity;
            var minEdges = new List<Vertex>();
            Vertex minE = null;
            foreach (var e in currentVertex.GetEdges())
            {
                float edgeDistance = e.GetDistance();
                // Debug.Log($"e: {e}, dist: {e.distance}, minD: {minD}, MinE: {minE}");
                if (edgeDistance>currentVertex.GetDistance())
                {
                    if (edgeDistance < minDistance)
                    {
                        minDistance = edgeDistance;
                        minEdges.Clear();
                        minEdges.Add(e);
                    }
                    else if (edgeDistance.Equals(minDistance))
                    {
                        minEdges.Add(e);
                    }
                }
            }
            if(minEdges.Count>0)
            {
                minE = minEdges[Random.Range(0, minEdges.Count - 1)];
                // minE = minEdges[randomState.Next(0, minEdges.Count)];
                if (minEdges.Contains(goalVertex)) minE = goalVertex;
            }
            currentVertex = minE;
            if (minE != null)
            {
                path.Add(minE);
            }
            
            if (currentVertex==null)
            {
                Debug.Log($"No valid path found from {startVertex} to {goalVertex}");
                break;
            }
        }
        return path;
    }

    private Vertex GetNearestVertex(Vector3 position)
    {
        float sqrDistance = Mathf.Infinity;
        Vertex closestV = null;
        foreach (var v in vertices)
        {
            float d = Vector3.SqrMagnitude(position - v.GetPosition());
            if (d < sqrDistance)
            {
                sqrDistance = d;
                closestV = v;
            }
        }
        return closestV;
    }
    
    public void SaveVertices()
    {
        WeightSaver.SerializeVertices(vertices, vertexSaveName);
    }

    public void LoadVertices()
    {
        if (vertices==null) return;
        if (vertices.Count == 0) return;

        WeightSaver.DeserializeVertices(out var vertexInfoDict, vertexLoadName);
        
        foreach (var keyValuePair in vertexInfoDict)
        {
            var index = keyValuePair.Key;
            if (index >= vertices.Count) break;
            if (!keyValuePair.Value.name.Equals(vertices[index].name))
            {
                Debug.LogError($"Name of loaded vertex does not match stored vertex at index {index}");
                continue;
            }
            var info = keyValuePair.Value;
            vertices[index].SetPosition(info.GetVectorPosition());
            foreach (var edgeWeight in info.edgeWeights)
            {
                vertices[index].SetEdgeWeight(edgeWeight.Key,edgeWeight.Value);
            }
        }
        Debug.Log("Vertices successfully loaded");
    }

    public static void DrawLine(Vector3 p1, Vector3 p2, Color color, float width)
    {
        Gizmos.color = color;
        int count = 1 + Mathf.CeilToInt(width); // how many lines are needed.
        if (count == 1)
        {
            Gizmos.DrawLine(p1, p2);
        }
        else
        {
            Camera c = Camera.current;
            if (c == null)
            {
                Debug.LogError("Camera.current is null");
                return;
            }
            var scp1 = c.WorldToScreenPoint(p1);
            var scp2 = c.WorldToScreenPoint(p2);
 
            Vector3 v1 = (scp2 - scp1).normalized; // line direction
            Vector3 n = Vector3.Cross(v1, Vector3.forward); // normal vector
 
            for (int i = 0; i < count; i++)
            {
                Vector3 o = 0.99f * n * width * ((float)i / (count - 1) - 0.5f);
                Vector3 origin = c.ScreenToWorldPoint(scp1 + o);
                Vector3 destiny = c.ScreenToWorldPoint(scp2 + o);
                Gizmos.DrawLine(origin, destiny);
            }
        }
        float triangleSide = 0.5f;
        var dir = (p2 - p1).normalized;
        var cross = Vector3.Cross(dir, Vector3.up);
        var right = p2 + (cross - dir) * triangleSide;
        var left = p2 + (-cross - dir) * triangleSide;
        Gizmos.DrawLine(p2,right);
        Gizmos.DrawLine(right,left);
        Gizmos.DrawLine(left,p2);
    }
    
#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (!showGizmos) return;

        foreach (var v in vertices)
        {
            if (!v) continue;
            var p1 = v.GetPosition();
            var edges = v.GetEdges();
            var weights = v.GetEdgeWeights();
            if (weights.Count == 0) continue;
            for (int i = 0; i < edges.Count; i++)
            {
                var p2 = edges[i].GetPosition();
                var w = weights[i];
                var thickness = 5f*w.GetStdDev()*2f; // Double to normalize maximum std dev of 0.5
                var color = Color.Lerp(Color.blue, Color.red, w.GetMean());
                color.a = 0.5f;
                DrawLine(p1,p2,color,thickness);
            }
            
        }
        
        // Drawn the shortest path according to start and goal positions.
        if (!shortestPath.Equals(null))
        {
            if (shortestPath.Count > 0)
            {
                Vertex start = shortestPath[0];
                if (start != null)
                {
                    Gizmos.color = Color.magenta;
                    Gizmos.DrawWireCube(start.GetPosition(),Vector3.one*0.15f);
                }
                Vertex goal = shortestPath[^1];
                if (goal != null)
                {
                    Gizmos.color = Color.green;
                    Gizmos.DrawWireCube(goal.GetPosition(),Vector3.one*0.15f);
                }
                for (int i = 0; i < shortestPath.Count-1; i++)
                {
                    var u = shortestPath[i];
                    var v = shortestPath[i + 1];
                    Gizmos.color = Color.green;
                    if(u!=null && v!= null) Gizmos.DrawLine(u.GetPosition(),v.GetPosition());
                }
            }
        }
    }
#endif
    
}
