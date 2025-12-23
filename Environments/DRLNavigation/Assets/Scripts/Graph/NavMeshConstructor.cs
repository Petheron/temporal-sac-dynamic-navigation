using System;
using System.Collections;
using System.Threading.Tasks;
using Unity.AI.Navigation;
using Unity.MLAgents;
using UnityEngine;

public class NavMeshConstructor : MonoBehaviour
{
    public VertexSamplingManager vertexSamplingManager;
    private NavMeshSurface surface;
    private Coroutine navmeshCoroutine;
    private AsyncOperation navMeshOperation;
    private float cachedTimeScale;

    public void BuildNavMesh()
    {
        if(!surface) surface = GetComponent<NavMeshSurface>();
        if (!surface) return;

        if (navmeshCoroutine != null)
        {
            StopCoroutine(navmeshCoroutine);
            navmeshCoroutine = null;
        }
        
        navmeshCoroutine = StartCoroutine(nameof(BuildAsyncNavMesh));
    }

    private IEnumerator BuildAsyncNavMesh()
    {
        if(!surface) surface = GetComponent<NavMeshSurface>();
        if (!surface) yield break;

        // if (!surface.navMeshData)
        // {
        //     surface.BuildNavMesh();
        //     yield return new WaitUntil(() => surface.navMeshData != null);
        // }
        //
        // // If there is an existing update operation, wait until it is complete before continuing.
        // if (navMeshOperation != null)
        // {
        //     // use this method as cannot yield an AsyncOperation more than once.
        //     yield return new WaitUntil(() => navMeshOperation.isDone);
        //     navMeshOperation = null;
        // }
        //
        // // start new update operation
        // navMeshOperation = surface.UpdateNavMesh(surface.navMeshData);
        //
        // // this should wait until it is complete.
        // yield return navMeshOperation;
        
        surface.BuildNavMesh();
        yield return new WaitUntil(() => surface.navMeshData != null);
        
        // Debug.LogWarning("NavMesh Built");
        
        // Now that the update/build is complete, we can construct the digraph.
        yield return StartCoroutine(nameof(BuildGraph));
        // navMeshOperation = null;
        
        // Debug.LogWarning("Vertex Graph Built");
        
        // Notify agent spawner that graph has been made and that agents can be spawned/reset.
        vertexSamplingManager.InvokeOnGraphMade();
    }

    private IEnumerator BuildGraph()
    {
        if (vertexSamplingManager)
        {
            vertexSamplingManager.CreateGraph();
            vertexSamplingManager.SetVertexSaveFileName(gameObject.name + "Vertices");
        }

        yield return null;
    }
}
