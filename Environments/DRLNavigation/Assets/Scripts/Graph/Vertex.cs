using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;
using Random = UnityEngine.Random;


public class Vertex : MonoBehaviour
{
    private List<Vertex> edges;
    private float minDistance = 1f;
    private float maxDistance = 10f;
    private float initialWeight = 100f;
    /// <summary>
    /// Cumulative distance for Djikestra's to this vertex.
    /// </summary>
    private float distance = Mathf.Infinity;
    // private Vector3 defaultPosition;
    private Dynamic dynamic;

    private List<RunningMeanStdDev> runningEdgeWeights;
    
    public Vector3 GetPosition() => transform.position;
    public List<Vertex> GetEdges() => edges;
    public Transform GetParent() => transform.parent;

    private bool isDynamic = false;

    public Vertex()
    {
        edges = new List<Vertex>();
        // edgeWeights = new List<float>();
        runningEdgeWeights = new List<RunningMeanStdDev>();

    }

    public void AddEdge(Vertex toVertex)
    {
        if (toVertex.Equals(this)) return;
        if (edges.Contains(toVertex))
        {
            return;
        }

        if (!IsValidEdge(toVertex)) return;
        
        edges.Add(toVertex);
        // edgeWeights.Add(initialWeight);
        runningEdgeWeights.Add(new RunningMeanStdDev());
    } 

    /// <summary>
    /// Remove the edge to the provided vertex from this vertex.
    /// </summary>
    /// <param name="toVertex">The potential edge vertex</param>
    public void RemoveEdge(Vertex toVertex)
    {
        int idx = edges.IndexOf(toVertex);
        if (idx == -1) return;
        edges.RemoveAt(idx);
        // edgeWeights.RemoveAt(idx);
        runningEdgeWeights.RemoveAt(idx);
        
    }

    /// <summary>
    /// Randomly sample an edge for this vertex. Returns null if no edges are found.
    /// </summary>
    /// <returns></returns>
    public Vertex SampleEdge()
    {
        if (NumEdges() == 0)
        {
            return null;
        }
        int edgeIndex = Random.Range(0, NumEdges());
        return edges[edgeIndex];
    }

    public Vertex GetEdgeVertex(int index)
    {
        Debug.Assert(index>=0 && index < NumEdges());
        return edges[index];
    }
    
    /// <summary>
    /// Update the validity of the edge from this vertex to the given one.
    /// </summary>
    /// <param name="toVertex"></param>
    public void UpdateEdgeValidity(Vertex toVertex)
    {
        if (toVertex.Equals(this)) return;
        RemoveEdge(toVertex);
        if (IsValidEdge(toVertex)) AddEdge(toVertex);
    }
    
    /// <summary>
    /// Prune edges by their distance, keeping only the k nearest neighbours.
    /// </summary>
    /// <param name="kNeighbours">Number of neighbouring edges to keep.</param>
    public void PruneEdges(int kNeighbours)
    {
        Debug.Assert(kNeighbours>=0);
        int numEdges = NumEdges();
        if (numEdges <= kNeighbours) return;
        var sortedEdges = edges;
        QuickSort(ref sortedEdges, 0, edges.Count-1);
        for (int i = numEdges-1; i >= kNeighbours; i--)
        {
            RemoveEdge(sortedEdges[i]);
        }
    }

    /// <summary>
    /// Update running edge weight with the newly calculated weight.
    /// </summary>
    /// <param name="edge"></param>
    /// <param name="newWeight"></param>
    public void UpdateEdgeWeight(Vertex edge, float newWeight)
    {
        int edgeIndex = edges.IndexOf(edge);
        if (edgeIndex == -1) return;
        runningEdgeWeights[edgeIndex].Update(newWeight);
    }

    public float SampleEdgeWeight(Vertex edge)
    {
        int edgeIndex = edges.IndexOf(edge);
        if (edgeIndex == -1) return initialWeight;
        return runningEdgeWeights[edgeIndex].SampleValue(0f,initialWeight);
    }

    /// <summary>
    /// Perform a Quick Sort on a given list of vertices.
    /// </summary>
    /// <param name="list">The list of vertices to sort in-place</param>
    /// <param name="low">Value to start sorting list from</param>
    /// <param name="high">Highest value to sort list</param>
    private void QuickSort(ref List<Vertex> list, int low, int high)
    {
        if (low < high)
        {
            int p = Partition(ref list, low, high);
            QuickSort(ref list, low, p-1);
            QuickSort(ref list, p+1, high);
        }
    }
    
    /// <summary>
    /// Sorts a partition of the Quicksort
    /// </summary>
    /// <param name="list"></param>
    /// <param name="low"></param>
    /// <param name="high"></param>
    /// <returns></returns>
    private int Partition(ref List<Vertex> list, int low, int high)
    {
        Vertex pivotVertex = list[high];
        float pivot = Vector3.Distance(pivotVertex.GetPosition(), GetPosition());
        int part = low - 1;

        for (int i = low; i < high; i++)
        {
            if (Vector3.Distance(list[i].GetPosition(), GetPosition()) < pivot)
            {
                part++;
                (list[i], list[part]) = (list[part], list[i]);
            }
        }

        list[high] = list[part + 1];
        list[part + 1] = pivotVertex;
        return part + 1;
    }

    /// <summary>
    /// Set the position of vertex.
    /// </summary>
    /// <param name="pos"></param>
    public void SetPosition(Vector3 pos)
    {
        // defaultPosition = pos;
        transform.position = pos;
    }
    
    /// <summary>
    /// Apply an offset to the default position of the vertex.
    /// Used for dynamic position updates.
    /// </summary>
    /// <param name="offset"></param>
    public void OffsetVertexPosition(Vector3 offset)
    {
        if (!isDynamic) return;
        // transform.position = defaultPosition + offset;
    }

    public void SetDynamic()
    {
        if (!GetParent()) return;
        if(!dynamic) dynamic = GetParent().GetComponent<Dynamic>();
        if (!dynamic) return;
        isDynamic = dynamic.isDynamic;
        // dynamic.OnPositionUpdate += OffsetVertexPosition;
    }

    public void SetDistanceConditions(float min, float max)
    {
        minDistance = min;
        maxDistance = max;
    }

    public void SetInitialWeight(float w)
    {
        initialWeight = w;
    }

    private bool IsValidEdge(Vertex potentialNode)
    {
        float d = Vector3.Distance(potentialNode.GetPosition(), GetPosition());
        return d >= minDistance && d<=maxDistance;
    }

    public int NumEdges()
    {
        return edges.Count;
    }

    public float GetDistance()
    {
        return distance;
    }

    public void SetDistance(float newDistance)
    {
        distance = newDistance;
    }

    public List<RunningMeanStdDev> GetEdgeWeights()
    {
        return runningEdgeWeights;
    }

    public void SetEdgeWeight(int index, RunningMeanStdDev newWeight)
    {
        if (index < 0 || index >= NumEdges()) return;
        runningEdgeWeights[index] = newWeight;
    }

    private void OnDisable()
    {
        // if(dynamic) dynamic.OnPositionUpdate -= OffsetVertexPosition;
    }
}
