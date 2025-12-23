using TMPro;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.Serialization;

public class AcademyStepCountUI : MonoBehaviour
{
    public TMP_Text academyStepCountText;

    private void Awake()
    {
        Academy.Instance.AgentPreStep += SetStepCountText;
    }

    private void SetStepCountText(int stepCount)
    {
        if (!academyStepCountText) return;
        academyStepCountText.text = $"StepCount: {stepCount}";
    }
}
