"""Numerical and training-schedule checks for the packed selector MLP."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from mmdd_stage2.column_batching import score_records
from mmdd_stage2.column_training import column_loss
from mmdd_stage2.verifier import CandidateColumnScorer


@pytest.mark.parametrize('head_type', ['linear', 'mlp'])
@pytest.mark.parametrize('training', [False, True])
def test_packed_logits_loss_gradients_updates_and_dropout_schedule(head_type, training):
    torch.set_num_threads(4)
    torch.manual_seed(73)
    records = [{'open_states': torch.randn(n, 16), 'close_states': torch.randn(n, 16),
                'candidate_column_indices': list(range(3, 3+n))} for n in (1, 7, 2, 13, 4)]
    scalar = CandidateColumnScorer(16, head_type=head_type).train(training)
    batched = copy.deepcopy(scalar)
    results, masks = [], []
    for model, execution in ((scalar, 'scalar'), (batched, 'batched')):
        observed = []
        handle = (model.weight[2].register_forward_hook(lambda module, args, output: observed.append(output != 0))
                  if head_type == 'mlp' else None)
        torch.manual_seed(19)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        logits = score_records(model, records, execution=execution)
        rng = torch.get_rng_state().clone()
        losses = [column_loss(scores, record['candidate_column_indices'],
                              [record['candidate_column_indices'][0], record['candidate_column_indices'][-1]])
                  for scores, record in zip(logits, records)]
        weights = torch.tensor([.3, 2., .5, 1.7, .8])
        loss = (torch.stack(losses)*weights).sum()/len(records)
        loss.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        results.append((logits, loss, gradients, rng))
        masks.append(observed)
        if handle is not None:
            handle.remove()
    for a, b in zip(results[0][0], results[1][0]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=3e-7)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-5, atol=3e-7)
    for a, b in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=5e-7)
    assert torch.equal(results[0][3], results[1][3])
    assert len(masks[0]) == len(masks[1])
    assert all(torch.equal(a, b) for a, b in zip(*masks))
    output_bias = 'weight.bias' if head_type == 'linear' else 'weight.3.bias'
    for (name, a), b in zip(scalar.named_parameters(), batched.parameters()):
        if name == output_bias:
            # This scalar adds the same constant to every column. Its exact
            # set-mass gradient is zero, but Adam amplifies FP32 residuals.
            # Check its gradient above and the resulting probabilities below.
            continue
        torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)
    scalar.eval()
    batched.eval()
    for a, b in zip(score_records(scalar, records), score_records(batched, records)):
        torch.testing.assert_close(a.softmax(0), b.softmax(0), rtol=1e-5, atol=1e-6)


def test_packed_keeps_empty_input_and_rejects_misaligned_markers():
    model = CandidateColumnScorer(4)
    assert score_records(model, []) == []
    with pytest.raises(ValueError, match='equal shapes'):
        score_records(model, [{'open_states': torch.zeros(2, 4), 'close_states': torch.zeros(3, 4)}])
    with pytest.raises(ValueError, match='Unknown head execution'):
        score_records(model, [], execution='pruned')


def test_missed_gold_still_fails_instead_of_dropping_training_pair():
    model = CandidateColumnScorer(4)
    records = [{'open_states': torch.zeros(2, 4), 'close_states': torch.ones(2, 4)}]
    logits = score_records(model, records)[0]
    with pytest.raises(ValueError, match='no mapped candidate'):
        column_loss(logits, [7, 11], [13])
