Shader "OGM/PointSurface"
{
    SubShader
    {
        CGPROGRAM
        #pragma surface ConfigureSurface Standard fullforwardshadows
        #pragma target 3.0

        float4 _ColorA, _ColorB; 

        struct Input
        {
            float3 worldPos;
        };
        

        void ConfigureSurface (Input input, inout SurfaceOutputStandard surface)
        {
            surface.Albedo = _ColorA;
        }
        ENDCG
    }
    
    Fallback "Diffuse"
}