from __future__ import annotations

import logging
from pathlib import Path

log_dir = Path(__file__).parents[1].resolve() / 'log'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / 'autoquake.log',
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
from .associator import GaMMA
from .focal import GAfocal
from .magnitude import Magnitude
from .picker import PhaseNet
from .polarity import DitingMotion
from .relocator import H3DD


def run_autoquake(
    picker: PhaseNet | None = None,
    associator: GaMMA | None = None,
    relocator: H3DD | None = None,
    pz_dir: Path | None = None,
    use_polarity=True,
    use_magnitude=True,
    mag_processes=30,
    use_focal=True,
    type_judge=None,
    station_file: Path | None = None,
    result_dir: Path | None = None,
    sac_parent_dir: Path | None = None,
    h5_parent_dir: Path | None = None,
):
    """## Run autoquake pipeline.

    ### Args
        - picker: PhaseNet object.
        - associator: GaMMA object.
        - relocator: H3DD object.
        - pz_dir: Path to the directory containing the poles and zeros files.
        - use_polarity: Whether to use polarity.
        - use_magnitude: Whether to use magnitude.
        - mag_processes: Number of processes to use for magnitude calculation.
        - use_focal: Whether to use focal mechanism.
        - type_judge: function for judging the type of station, use for focal mechanism.
                      Using None as default, which means using the second character of station to judge the type of station.
    """
    # First part, need more configuration
    ## Run PhaseNet
    if picker is not None:
        logging.info('PhaseNet start')
        picker.run_predict()
        logging.info('PhaseNet end')
        data_parent_dir = Path(picker.data_parent_dir)
        result_path = Path(picker.result_path)

    ## Run GaMMA
    if associator is not None:
        logging.info('GaMMA start')
        associator.run_predict()
        logging.info('GaMMA end')
        gamma_picks = associator.picks
        station = associator.station

    ## Run H3DD
    if relocator is not None:
        logging.info('H3DD start')
        relocator.run_h3dd()
        logging.info('H3DD end')
        gamma_reorder_event = relocator.reorder_event
        dout_file = relocator.dout

    # Second part, arguments are path, created inside the function.
    ## Polarity. Run DitingMotion
    if use_polarity:
        logging.info('DitingMotion start')
        if picker is not None:
            if picker.format == 'SAC':
                dt_focal = DitingMotion(
                    gamma_picks=gamma_picks,
                    sac_parent_dir=data_parent_dir,
                    output_dir=result_path,
                    type_judge=type_judge,
                )
            elif picker.format == 'h5':
                dt_focal = DitingMotion(
                    gamma_picks=gamma_picks,
                    h5_parent_dir=data_parent_dir,
                    output_dir=result_path,
                    type_judge=type_judge,
                )
            dt_focal.run_parallel_predict(processes=3)
            polarity_picks = dt_focal.picks

    ## Magnitude. Run Magnitude
    if use_magnitude:
        logging.info('Magnitude start')
        if pz_dir is None:
            raise ValueError('pz_dir is None, please provide pz_dir')
        mag = Magnitude(
            dout_file=dout_file,
            station=station,
            sac_parent_dir=data_parent_dir,
            pz_dir=pz_dir,
            output_dir=result_path,
        )
        mag.run_mag(processes=mag_processes)
        mag_events = mag.events
        mag_picks = mag.picks

    ## Focal. Run GAfocal
    if use_polarity and use_focal:
        logging.info('GAfocal start')
        output_dir = Path(__file__).parents[1].resolve() / 'GAfocal'
        dout_file_name = dout_file.name
        if use_magnitude:
            logging.info('Format converting with pol and mag...')
            dout_file_name = H3DD.pol_mag_to_dout(
                ori_dout=dout_file,
                gamma_reorder_event=gamma_reorder_event,
                polarity_picks=polarity_picks,
                magnitude_events=mag_events,
                magnitude_picks=mag_picks,
                output_path=output_dir,
            )
            logging.info('Format converting over.')
        else:
            logging.info('Format converting with pol...')
            dout_file_name = H3DD.pol_to_dout(
                ori_dout=dout_file,
                gamma_reorder_event=gamma_reorder_event,
                polarity_picks=polarity_picks,
                output_path=output_dir,
            )
            logging.info('Format converting over.')
        gafocal = GAfocal(dout_file_name=dout_file_name, result_path=result_path)
        gafocal.run()
        logging.info('GAfocal end')
