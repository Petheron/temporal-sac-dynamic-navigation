using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;


public class WeightSaver
{
    /// <summary>
    /// Vertex information to be serialized to JSON. Used instead of serializing vertex directly due recursive paths in GameObject class.
    /// </summary>
    [Serializable]
    public class VertexInformation
    {
        public string name;
        public float[] position;
        public Dictionary<int, RunningMeanStdDev> edgeWeights;

        public VertexInformation(Vertex v)
        {
            edgeWeights = new Dictionary<int, RunningMeanStdDev>();
            ProcessVertex(v);
        }

        /// <summary>
        /// Extracts vertex information for a provided vertex.
        /// </summary>
        /// <param name="v">Vertex to serialize</param>
        public void ProcessVertex(Vertex v)
        {
            if (!v) return;
            name = v.name;
            Vector3 pos = v.GetPosition();
            position = new []{pos[0],pos[1],pos[2]};
            var edges = v.GetEdges();
            var weights = v.GetEdgeWeights();
            for (int i = 0; i < edges.Count; i++)
            {
                edgeWeights.Add(i, weights[i]);
            }
        }

        /// <summary>
        /// Get Vector3 form of saved float[] position.
        /// </summary>
        /// <returns>Global position of vertex</returns>
        public Vector3 GetVectorPosition()
        {
            return new Vector3(position[0], position[1], position[2]);
        }
    }

    /// <summary>
    /// Serialize vertices to JSON file in Resources/JSON/. 
    /// </summary>
    /// <param name="vertices">Vertices to serialize</param>
    /// <param name="fileName">Name of JSON file.</param>
    public static void SerializeVertices(List<Vertex> vertices, string fileName = "Vertices")
    {
        Dictionary<int,VertexInformation> vertexInformations = new Dictionary<int, VertexInformation>();
        for (int i = 0; i < vertices.Count; i++)
        {
            vertexInformations.Add(i, new VertexInformation(vertices[i]));
        }

        JsonSerializer serializer = new JsonSerializer();

        var path = GetUniqueSavePath(fileName);

        using StreamWriter sw = new StreamWriter(path);
        using JsonWriter writer = new JsonTextWriter(sw);
        serializer.Serialize(writer,vertexInformations);
    }
    
    /// <summary>
    /// Loads JSON serialized vertices from file in Resources/JSON/. Used for vertex weight loading.
    /// </summary>
    /// <param name="vertexInformations">Output dictionary of vertex information</param>
    /// <param name="fileName">JSON filename</param>
    public static void DeserializeVertices(out Dictionary<int,VertexInformation> vertexInformations, string fileName)
    {
        JsonSerializer serializer = new JsonSerializer();
        using StreamReader sr = new StreamReader(Application.dataPath+$"/Resources/JSON/{fileName}.txt");
        using JsonTextReader reader = new JsonTextReader(sr);
        vertexInformations=serializer.Deserialize<Dictionary<int,VertexInformation>>(reader);
    }

    private static String GetUniqueSavePath(String fileName)
    {
        String path = Application.dataPath + "/Resources/JSON/";
        var file = path + fileName;
        var ext = ".txt";
        while (File.Exists(file+ext))
        {
            file += "~";
        }

        return file+ext;
    }
    
    
}
