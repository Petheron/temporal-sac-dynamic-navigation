using UnityEngine;

public class PythagoreanAngle
{
    public int a;
    public int b;
    public int c;

    public PythagoreanAngle(float angle, float eps)
    {
        SetTriple(angle, eps);
    }

    /// <summary>
    /// Approximate a euclidean angle with a pythagorean triple.
    /// https://math.stackexchange.com/questions/7143/is-it-possible-to-get-arbitrarily-near-any-acute-angle-with-pythagorean-triangle
    /// </summary>
    /// <param name="angle">Angle to approximate (in degrees).</param>
    /// <param name="eps">Approximation error for given angle.</param>
    private void SetTriple(float angle, float eps)
    {
        angle = Mathf.Repeat(angle, 90f);
        angle *= Mathf.Deg2Rad;
        eps *= Mathf.Deg2Rad;
        int N = 1<<24; //2^24
            
        // t is value to approx with Farey Algorithm. Derived for pythagorean triangle with sides (u^2-v^2, 2uv, u^2+v^2), u>v.
        // Small angle approximation (sin(x)-->x, cos(x)--> 1-x*x/2, tan(x)-->x) if necessary
        float t = UseSmallAngleApproximation(angle) ? angle/2f : (1f - Mathf.Cos(angle)) / Mathf.Sin(angle);
        float tError = UseSmallAngleApproximation(eps) ? eps / 2f : (1f - Mathf.Cos(eps)) / Mathf.Sin(eps);
            
        Vector2Int nm = FareyAlgorithm(t, N, tError);
        int n = nm[0];
        int m = nm[1];
            
        a = m * m - n * n;
        b = 2 * n * m;
        c = n * n + m * m;
    }

    public float Angle()
    {
        return Mathf.Atan2(b, a) * Mathf.Rad2Deg;
    }

    public float CosTerm()
    {
        return Mathf.Cos(Mathf.Atan2(b, a));
    }

    public override string ToString()
    {
        return $"({a},{b},{c})";
    }
    
    /// <summary>
    /// Is a given angle (in radians) small enough to be considered "small"?
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private static bool UseSmallAngleApproximation(float x)
    {
        return Mathf.Abs(Mathf.Cos(x)-1f) < Mathf.Epsilon;

    }

    /// <summary>
    /// Use Farey Sequence to find two integers that approximate a given float between 0-1.
    /// https://www.johndcook.com/blog/2010/10/22/approximations-with-pythagorean-triangles/
    /// </summary>
    /// <param name="x">Value to approximate.</param>
    /// <param name="N">Maximum side length of pythagorean triangle approximation.</param>
    /// <param name="eps">Accuracy of approximation.</param>
    /// <returns>Pythagorean Sides (n,m)</returns>
    private static Vector2Int FareyAlgorithm(float x, int N=100, float eps=0.01f)
    {
        int a = 0;
        int b = 1;
        int c = 1;
        int d = 1;

        float mediant = 0f;

        while (b<=N && d<=N)
        {
            mediant = (float)(a + c) / (b + d);
            
            if (Mathf.Abs(mediant - x) <= eps)
            {
                if (b + d <= N)
                {
                    return new Vector2Int(a + c, b + d);
                }
                if (d > b)
                {
                    return new Vector2Int(c, d);
                }
                return new Vector2Int(a, b);
            }
            if (x > mediant)
            {
                a += c;
                b += d;
            }
            else
            {
                c += a;
                d += b;
            }
        }

        if (b > N)
        {
            return new Vector2Int(c, d);
        }

        return new Vector2Int(a, b);
    }
}