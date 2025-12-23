using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Damage : MonoBehaviour
{
    public int damageValue = 1;
    
    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("Agent"))
        {
            var health = other.gameObject.GetComponent<Health>();
            if(health) health.TakeDamage(damageValue);
        }
    }
}
