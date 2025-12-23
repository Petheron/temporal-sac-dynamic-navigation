using UnityEngine;

public class MutatorComponent:MonoBehaviour
{
    public MutatorType mutatorType;
    private Mutator mutator;

    private void Awake()
    {
        mutator = new Mutator(mutatorType);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.isTrigger) return;
        var move = other.GetComponent<ModularMovement>();
        if (move)
        {
            move.SetMutator(mutator);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.isTrigger) return;
        var move = other.GetComponent<ModularMovement>();
        if (move)
        {
            if(move.currentMutatorType==mutatorType && move.newMutator==null) move.SetMutator(MutatorType.None);
            move.ApplyNewMutator();
        }
    }
}
