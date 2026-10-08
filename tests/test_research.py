import numpy as np
import pytest

def test_event_matching_requires_class_and_one_to_one():
    from research.protocol import event_metrics
    truth=[(1,0.,1.),(2,2.,3.)]
    pred=[(1,.1,1.1),(1,.2,1.),(3,2.,3.)]
    m=event_metrics(truth,pred,background_seconds=10)
    assert m['tp']==1 and m['fp']==2 and m['fn']==1
    assert m['duplicates']==1 and m['event_f1']==pytest.approx(.4)

def test_split_people_disjoint_and_missing_retained():
    from research.protocol import split_people, frame_labels
    s=split_people(['A','B','C','D','E','F','G','H','I','J'])
    assert not set(s['train']) & set(s['test'])
    assert not set(s['train']) & set(s['val'])
    assert sum(map(len,s.values()))==10
    assert np.array_equal(frame_labels([1,4,7],[(2,4,7)]),[0,2,2])

def test_causal_model_does_not_read_future():
    import torch
    from gesture_system.temporal import CausalNet
    torch.manual_seed(2)
    m=CausalNet(49,14,24).eval()
    x=torch.randn(2,20,49); y=x.clone(); y[:,10:]+=10
    with torch.no_grad():
        a=m(x)[0]; b=m(y)[0]
    torch.testing.assert_close(a[:,:10],b[:,:10])

def test_predictor_missing_and_bounded_history(tmp_path):
    import torch
    from gesture_system.temporal import CausalNet,TemporalPredictor
    m=CausalNet(49,14,24)
    p=tmp_path/'model.pt'
    torch.save({'state_dict':m.state_dict(),'hidden':24,'labels':['D0X','B0A','B0B']+[f'G{i:02}' for i in range(1,12)],'mean':torch.zeros(49),'std':torch.ones(49),'method':'joint','history':20},p)
    model=TemporalPredictor(p)
    for _ in range(40): model.update(np.ones(48,dtype=np.float32),True)
    assert len(model.history)==20
    assert model.update(np.ones(48,dtype=np.float32),False)==('none',0.,0.)

def test_confirmation_is_causal_and_preserves_background():
    from research.run import confirm_labels,boundary_targets
    assert np.array_equal(confirm_labels([0,1,1,1,0,2,2],2),[0,0,1,1,0,0,2])
    b=boundary_targets(np.array([0,1,1,2,2,0]))
    assert np.array_equal(b[:,0],[0,1,1,1,1,0])
    assert np.array_equal(b[:,1],[0,1,0,1,0,0])
    assert np.array_equal(b[:,2],[0,0,1,0,1,0])

def test_two_stage_predictor_uses_independent_detector(tmp_path):
    import torch
    from gesture_system.temporal import CausalNet,TemporalPredictor
    from research.protocol import LABELS
    m=CausalNet(49,14,48,method='two_stage');det=CausalNet(49,14,24)
    with torch.no_grad():det.boundary.weight.zero_();det.boundary.bias.fill_(-100)
    p=tmp_path/'two.pt'
    torch.save(dict(state_dict=m.state_dict(),detector_state_dict=det.state_dict(),hidden=48,method='two_stage',labels=LABELS,mean=torch.zeros(49),std=torch.ones(49),history=30),p)
    model=TemporalPredictor(p)
    assert model.update(np.zeros(48),True)[2]<1e-6

def test_fragment_without_matched_event_is_not_duplicate():
    from research.protocol import event_metrics
    result=event_metrics([(1,0.,2.)],[(1,0.,.1)],10.)
    assert result['tp']==0 and result['duplicates']==0

def test_unannotated_tail_is_not_invented_background():
    from research.protocol import frame_labels
    assert np.array_equal(frame_labels([1,4,7],[(0,1,2),(2,3,5)],unknown=-1),[0,2,-1])

def test_fewshot_quantiles_record_actual_p95():
    from research.fewshot import latency_statistics
    result=latency_statistics([1,2,3,4,100])
    assert result['latency_p50_ms']==3
    assert result['latency_p95_ms']==pytest.approx(80.8)

def test_controller_decoder_does_not_retrigger_without_neutral():
    from research.decoder_ablation import conservative_decode
    times=np.arange(18)*.1
    raw=np.array([0,0,0,1,1,1,1,2,2,2,0,0,0,2,2,2,2,0])
    out=conservative_decode(raw,times,confirm=3,neutral_s=.3,cooldown_s=.5)
    assert np.array_equal(out,[0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,2,2,0])

def test_one_shot_command_matches_timestamp_not_interval():
    from research.decoder_ablation import command_metrics
    m=command_metrics([(1,0.,1.),(2,2.,3.)],[(1,.5),(1,.8),(3,2.5),(2,4.)],10)
    assert (m['tp'],m['fp'],m['fn'],m['duplicates'],m['background_fp'])==(1,3,1,1,1)
    assert m['command_precision']==.25 and m['command_recall']==.5
