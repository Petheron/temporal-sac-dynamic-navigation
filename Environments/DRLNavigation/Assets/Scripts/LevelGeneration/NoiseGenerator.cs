using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;

namespace LevelGeneration
{
    public struct AdditionalNoiseParameters
    {
        public int seed;
        public float amplitude;
        public float frequency;
        public int iterations;
        public float persistence;
        public float seedFrequency;

        public AdditionalNoiseParameters(int seed, float amplitude, float frequency, int iterations, float persistence, float seedFrequency)
        {
            this.seed = seed;
            this.amplitude = amplitude;
            this.frequency = frequency;
            this.iterations = iterations;
            this.persistence = persistence;
            this.seedFrequency = seedFrequency;
        }
    }
    
    public class NoiseGenerator
    {
        protected byte[,,] grid;

        /// <summary>
        /// Generate grid of chunk with Simplex Noise. Method uses a 3D slice of a 4D noise sample (x,y,z,seed).
        /// </summary>
        /// <param name="sizeX">x-dimension of the grid.</param>
        /// <param name="sizeY">y-dimension of the grid.</param>
        /// <param name="sizeZ">z-dimension of the grid.</param>
        /// <param name="offset">The world space offset of the grid to ensure chunks alignment.</param>
        /// <param name="cellSize">Size of each cell of the chunk.</param>
        /// <param name="noiseFrequency">The frequency of the simplex noise. Higher value means features are closer together.</param>
        /// <param name="seed">Sampling seed for the noise.</param>
        /// <returns></returns>
        public virtual byte[,,] Generate(int sizeX, int sizeY, int sizeZ,  Vector3 offset, float cellSize, AdditionalNoiseParameters noiseParams)
        {
            grid = new byte[sizeX, sizeY, sizeZ];
            
            for (int i = 0; i < sizeX; i++)
            {
                for (int j = 0; j < sizeY; j++)
                {
                    for (int k = 0; k < sizeZ; k++)
                    {
                        float3 pos = new float3(i*cellSize, j*cellSize, k*cellSize) + new float3(offset);
                        // float3 seedOffset = new float3(seed,seed,seed);
                        // seedOffset *= 0.001f;
                        // pos *= noiseFrequency;
                        // // pos += seedOffset;
                        // float4 seededPos = new float4(pos.x, pos.y, pos.z, 0.001f*seed);
                        // // grid[i,j,k] = (byte) (noise.cnoise(pos) >= 0f ? 1: 0);
                        // float value = noise.snoise(seededPos);
                        float value = SumOctave(pos, noiseParams);
                        grid[i,j,k] = (byte) (value >= 0f ? 1: 0);
                    }
                }
            }
            return grid;
        }

        private float SumOctave(float3 pos, AdditionalNoiseParameters noiseParams)
        {
            float sumAmp = 0;
            float amp = noiseParams.amplitude;
            float freq = noiseParams.frequency;
            float output = 0;

            float high = 1f;
            float low = -1f;

            for (int i = 0; i < noiseParams.iterations; i++)
            {
                float4 input = new float4(pos * freq, noiseParams.seed*noiseParams.seedFrequency);
                output += noise.snoise(input)*amp;

                sumAmp += amp;
                amp *= noiseParams.persistence;
                freq *= 2;
            }

            // average and normalize
            output /= sumAmp;
            output = output * (high - low) / 2f + (high + low) / 2f;

            return output;

        }
    }
}