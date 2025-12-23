using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;
using Random = UnityEngine.Random;

public class Cannon : MonoBehaviour
{
    public Transform projectileSpawnTransform;
    public GameObject projectilePrefab;
    public float startDelay = 2f;
    public float fireRate=0.5f;

    private float timer;
    [SerializeField,ReadOnly]private float randomDelay;
    private bool CanFire => timer >= fireRate+randomDelay;
    private Coroutine firingCycle;

    private void Awake()
    {
        randomDelay = Random.Range(0f, startDelay);
    }

    private void Start()
    {
        Reset();
    }

    private void FixedUpdate()
    {
        if (CanFire)
        {
            Shoot();
            timer = randomDelay;
        }
        timer += Time.fixedDeltaTime;
    }

    private void Shoot()
    {
        Instantiate(projectilePrefab, projectileSpawnTransform.position, transform.rotation);
    }

    private void Reset()
    {
        timer = 0f;
    }

    
}
