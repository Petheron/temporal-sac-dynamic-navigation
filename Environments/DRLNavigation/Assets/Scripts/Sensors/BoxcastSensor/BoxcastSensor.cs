using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;
using Unity.MLAgents;
using Unity.VisualScripting.FullSerializer;

namespace Unity.MLAgents.Sensors
{   
    /// <summary>
    /// Boxcast sensor.
    /// Used to sample 3D occupancy map in vicinity of agent.
    /// </summary>
    public class BoxcastSensor : ISensor, IDisposable
    {
        private string name;
        private Vector3Int sampleSize;
        private ObservationSpec observationSpec;

        private SensorCompressionType compressionType;
        public IBoxcastPerception gridPerception;
        
        // Buffers
        private int[] perceptionBuffer;
        private int[] indicesBuffer;

        // Utility Constants Calculated on Init
        private int numCells;
        private int numChannels;

        public BoxcastSensor(
            string name,
            Vector3Int sampleSize,
            int numChannels = 1
        )
        {
            this.name = name;
            this.sampleSize = sampleSize;
            this.numChannels = numChannels;

            numCells = sampleSize.x * sampleSize.y * sampleSize.z;
            compressionType = SensorCompressionType.None;
            observationSpec = MakeBoxcastObservationSpec(sampleSize.y, sampleSize.z, sampleSize.x, numChannels);

            ResetPerceptionBuffer();
        }
        
        private ObservationSpec MakeBoxcastObservationSpec(int depth, int height, int width, int channels, ObservationType obsType = ObservationType.Default)
        {
            var dimProps = new InplaceArray<DimensionProperty>(
                DimensionProperty.TranslationalEquivariance,
                DimensionProperty.TranslationalEquivariance,
                DimensionProperty.TranslationalEquivariance,
                DimensionProperty.None
            );
            return new ObservationSpec(
                new InplaceArray<int>(depth, height, width, channels),
                dimProps,
                obsType
            );
        }

        internal int[] PerceptionBuffer
        {
            get { return perceptionBuffer; }
        }
        
        internal int[] IndicesBuffer
        {
            get { return indicesBuffer; }
        }

        public void Reset(){}

        private void ResetPerceptionBuffer()
        {
            if (perceptionBuffer != null)
            {
                Array.Clear(perceptionBuffer,0,perceptionBuffer.Length);
            }
            else
            {
                perceptionBuffer = new int[numCells*numChannels];
            }
            
            if (indicesBuffer != null)
            {
                Array.Clear(indicesBuffer,0,indicesBuffer.Length);
            }
            else
            {
                indicesBuffer = new int[numCells*numChannels];
            }
        }
        
        public string GetName()
        {
            return name;
        }

        public CompressionSpec GetCompressionSpec()
        {
            return new CompressionSpec(compressionType);
        }

        /// <summary>
        ///  Returns compressed observation for PNG. Required for ISensor.
        ///  Not implemented as PNG meaningless for 3D texture.
        /// </summary>
        /// <returns></returns>
        public byte[] GetCompressedObservation()
        {
            return new List<byte>().ToArray();
        }

        private bool IsDataNormalized()
        {
            return false;
        }

        private ProcessCollidersMethod GetProcessCollidersMethod()
        {
            return ProcessCollidersMethod.ProcessClosestColliders;
        }

        /// <summary>
        /// Collect data from detected object if a detectable tag is found.
        ///  If processing all colliders, copy the array from perception buffer to cell buffer for the relevant cell index.
        ///  Otherwise clear out cell data.
        /// Then store object data into cell data buffer, and copy into perception buffer. 
        /// </summary>
        /// <param name="detectedObject"></param>
        /// <param name="cellIndex"></param>
        public void ProcessSampleArray(int[] sampleArray)
        {
            // Profiler.BeginSample("BoxcastSensor.ProcessDetectedObjects");
            Array.Copy(sampleArray,0,perceptionBuffer,0,sampleArray.Length);
            // Profiler.EndSample();
        }

        public void ProcessIndicesArray(int[] indicesArray)
        {
            Array.Copy(indicesArray,0,indicesBuffer,0,indicesArray.Length);
        }

        public void Update()
        {
            ResetPerceptionBuffer();

            if (gridPerception != null)
            {
                gridPerception.Perceive();
            }
            
        }

        public ObservationSpec GetObservationSpec()
        {
            return observationSpec;
        }

        public int Write(ObservationWriter writer)
        {
            int index = 0;
            for (var d = 0; d < sampleSize.y; d++)
            {
                for(var h = 0; h < sampleSize.z; h++)
                {
                    for (var w = 0; w < sampleSize.x; w++)
                    {
                        for (var c = 0; c < numChannels; c++)
                        {
                            // Observation writer can only take (h,w,c) by default
                            // idx = c + Cx + CXz + CXZy
                            writer[d, h, w, c] = perceptionBuffer[index];
                            index++;
                        }
                    }
                }
            }

            return index;
        }

        public void Dispose()
        {
            // if (!ReferenceEquals(null, m_PerceptionTexture))
            // {
            //     if (Application.isEditor)
            //     {
            //         UnityEngine.Object.DestroyImmediate(m_PerceptionTexture);
            //     }
            //     else
            //     {
            //         UnityEngine.Object.Destroy(m_PerceptionTexture);
            //     }
            //     m_PerceptionTexture = null;
            // }
        }
    }
}




