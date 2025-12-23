using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrumblingPlatform : Dynamic
{
    public float crumbleTime = 1.0f;
    public float crumbleDuration = 1.0f;
    public float resetTime = 2.0f;

    private Renderer rend;

    [SerializeField] private float timer;
    private Coroutine coroutine;
    private Color color;
    private MaterialPropertyBlock block;
    private float startTime;
    

    private BoxCollider[] colliders;
    
    // Start is called before the first frame update
    protected override void Start()
    {
        base.Start();
        rend = GetComponent<Renderer>();
        color = rend.material.color;
        block = new MaterialPropertyBlock();
        colliders = GetComponents<BoxCollider>();
    }

    protected override void PerformDiscreteDynamics(Collider other)
    {
        base.PerformDiscreteDynamics(other);
        if (!other.CompareTag("Agent")) return;
        if (coroutine != null) return;
        coroutine = StartCoroutine(Crumble());
    }

    protected override void Test()
    {
        if (coroutine != null) return;
        coroutine = StartCoroutine(Crumble());
    }

    private IEnumerator Crumble()
    {
        // print("Begin");
        yield return new WaitForSeconds(crumbleTime);
        // print("Crumbling");
        
        StartCoroutine(Fade());
    }

    private IEnumerator Fade()
    {
        timer = crumbleDuration;
        // print("Fade start");
        while (timer>0)
        {
            timer -= Time.fixedDeltaTime;
            float alpha = timer / crumbleDuration;
            alpha = Mathf.Max(0f, alpha);
            SetAlpha(alpha);
            // LerpColor(fadeColor, color, alpha);
            yield return new WaitForFixedUpdate();
        }
        SubtractAllOccupancyNodes();
        ToggleColliders(false);
        timer = 0f;
        yield return new WaitForSeconds(resetTime);
        Appear();
        coroutine = null;
    }

    private void Appear()
    {
        SetAlpha(1.0f);
        // LerpColor(fadeColor, color, 1f);
        AddAllOccupancyNodes();
        ToggleColliders(true);
    }

    private void ToggleColliders(bool isEnabled)
    {
        foreach (var c in colliders)
        {
            c.enabled = isEnabled;
        }
    }

    private void LerpColor(Color a, Color b, float t)
    {
        Color l = Color.Lerp(a, b, t);
        block.SetColor("_BaseColor", l);
        rend.SetPropertyBlock(block,0);
    }

    private void SetAlpha(float alpha)
    {
        Color col = color;
        col.a = alpha;
        block.SetColor("_BaseColor", col);
        rend.SetPropertyBlock(block,0);
    }
}
