using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class RateText : MonoBehaviour
{
    public TrajectoryVisualizer visualizer;
    public EndCondition condition;
    [SerializeField] private TextMeshProUGUI tmp;
    private void Awake()
    {
        tmp = GetComponent<TextMeshProUGUI>();
        SetText();
        if (visualizer) visualizer.OnTrajComplete += SetText;
    }

    private void SetText()
    {
        if (!visualizer) return;
        if (!tmp) return;
        int value = Mathf.RoundToInt(visualizer.DRLTrajRates[(int) condition] * 100f);
        tmp.text = $"{value}%";
    }
}
