
public class ActionRepeater:ModularDecisionRequesterComponent
{
    protected override void Awake()
    {
        base.Awake();
        takeActionsBetweenDecisions = true;
    }

    public override void InitializeAgentProperties()
    {
        base.InitializeAgentProperties();
        decisionPeriod = modularAgentProperties.ACTION_REPETITION;
    }

    public override void InitializeEnvironmentParameters()
    {
        base.InitializeEnvironmentParameters();
        decisionPeriod = (int) environmentParameters.GetWithDefault("action_repetition", decisionPeriod);
    }
}
