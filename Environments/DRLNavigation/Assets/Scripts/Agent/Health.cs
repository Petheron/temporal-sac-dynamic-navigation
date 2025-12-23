using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Health : MonoBehaviour
{
    private int currentHealth;
    [SerializeField] private int maxHealth = 1;
    public bool IsAlive => currentHealth > 0;
    public event Action OnDeath;

    private void Awake()
    {
        Reset();
    }

    public void Reset()
    {
        currentHealth = maxHealth;
    }

    public void TakeDamage(int damage)
    {
        if (damage <= 0) return;
        if (!IsAlive) return;
        
        currentHealth -= damage;

        if (currentHealth > 0) return;
        
        currentHealth = 0;
        OnDeath?.Invoke();
    }
}
