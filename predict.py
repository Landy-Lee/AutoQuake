from pathlib import Path

from autoquake.associator import GaMMA
from autoquake.picker import PhaseNet
from autoquake.relocator import H3DD
from autoquake.scenarios import run_autoquake

if __name__ == '__main__':
    repo_dir = Path('/home/patrick/Work/AutoQuake/')
    sac_parent_dir = Path('/raid1/share/for_patrick/AutoQuake_testset/SAC')
    station_sac = Path('/raid1/share/for_patrick/AutoQuake_testset/station_seis.csv')
    pz_dir = Path('/raid1/share/for_patrick/AutoQuake_testset/PZ_dir/')
    result_path = repo_dir / 'results_test'
    h5_parent_dir = Path('/raid1/share/for_patrick/AutoQuake_testset/hdf5/')
    startdate = '20240402'
    enddate = '20240402'
    station_das = Path('/raid1/share/for_patrick/AutoQuake_testset/station_das.csv')
    model_3d = Path(
        '/home/patrick/Work/AutoQuake/H3DD/tomops_H14'
    )  # for relocation (h3dd)
    gamma_vel_model_1 = Path(
        '/home/patrick/Work/AutoQuake/vel_model/midas_vel.csv'
    )  # for gamma to compute travel time
    gamma_vel_model_2 = Path(
        '/home/patrick/Work/AutoQuake_pamicoding/GaMMA/Hualien_data_20240402/Hualien_1D.vel'
    )  # for gamma to compute travel time

    phasenet = PhaseNet(
        data_parent_dir=sac_parent_dir,
        start_ymd=startdate,
        end_ymd=enddate,
        result_path=result_path,
        format='SAC',
        model='phasenet',
        device='cpu',
    )

    gamma = GaMMA(
        station=station_sac,
        result_path=result_path,
        pickings=phasenet.picks,
        vel_model=gamma_vel_model_2,
    )

    h3dd = H3DD(
        gamma_event=gamma.events,
        gamma_picks=gamma.picks,
        station=gamma.station,
        model_3d=model_3d,
        event_name=result_path.name,
    )

    run_autoquake(picker=phasenet, associator=gamma, relocator=h3dd, pz_dir=pz_dir)
