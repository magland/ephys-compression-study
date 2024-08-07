import os
import shutil
import tempfile
import numpy as np
import lindi
import pynwb
import spikeinterface as si
from pynwb.ecephys import ElectricalSeries
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
from nwbextractors import NwbRecordingExtractor
from qfc.codecs import QFCCodec
from qfc import qfc_estimate_quant_scale_factor
from _scale_recording_if_float_type import _scale_recording_if_float_type
from make_float32_recording import make_float32_recording

QFCCodec.register_codec()


def ephys_compression_study():
    if not os.path.exists('data'):
        os.makedirs('data')

    # https://neurosift.app/?p=/nwb&url=https://api.dandiarchive.org/api/assets/2e6b590a-a2a4-4455-bb9b-45cc3d7d7cc0/download/&dandisetId=000463&dandisetVersion=draft
    url = "https://api.dandiarchive.org/api/assets/2e6b590a-a2a4-4455-bb9b-45cc3d7d7cc0/download/"
    electrical_series_name = 'ElectricalSeries'

    _create_nwb_lindi(h5_url=url, output_file="data/example.nwb.lindi")
    for cr in [0, 4, 8, 12, 16, 20, 24, 28, 50, 100]:
        print('\n\n\n\n\n\n')
        print(f'Compression ratio: {cr}')
        pre_fname = f'data/example_pre_{cr}.nwb.lindi'
        units_fname = f'data/example_units_{cr}.nwb.lindi'
        _create_ephys_pre(
            input_fname='data/example.nwb.lindi',
            electrical_series_name=electrical_series_name,
            output_fname=pre_fname,
            compression_type='qfc',
            compression_opts={
                'target_compression_ratio': cr,
                'compression_method': 'zlib',
                'zlib_level': 3,
                'zstd_level': 3
            }
        )
        _run_spike_sorting(
            input_fname=pre_fname,
            electrical_series_name=electrical_series_name + "_pre",
            output_fname=units_fname,
        )


def _create_nwb_lindi(*, h5_url: str, output_file: str):
    if os.path.exists(output_file):
        return
    cache = lindi.LocalCache()
    with lindi.LindiH5pyFile.from_hdf5_file(h5_url, local_cache=cache) as f:
        f.write_lindi_file(output_file)


def _create_ephys_pre(*, input_fname: str, electrical_series_name: str, output_fname: str, compression_type: str, compression_opts: dict):
    if os.path.exists(output_fname):
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        output_fname_tmp = tmpdir + '/output.nwb.lindi'
        shutil.copyfile(input_fname, output_fname_tmp)
        cache = lindi.LocalCache()
        with lindi.LindiH5pyFile.from_lindi_file(output_fname_tmp, mode="r+", local_cache=cache) as f:
            electrical_series_path = '/acquisition/' + electrical_series_name

            recording = NwbRecordingExtractor(
                h5py_file=f, electrical_series_path=electrical_series_path
            )

            num_frames = recording.get_num_frames()
            start_time_sec = 0
            # duration_sec = 300
            duration_sec = num_frames / recording.get_sampling_frequency()
            start_frame = int(start_time_sec * recording.get_sampling_frequency())
            end_frame = int(np.minimum(num_frames, (start_time_sec + duration_sec) * recording.get_sampling_frequency()))
            recording = recording.frame_slice(
                start_frame=start_frame,
                end_frame=end_frame
            )

            # bandpass filter
            freq_min = 300
            freq_max = 6000
            recording_filtered = spre.bandpass_filter(
                recording, freq_min=freq_min, freq_max=freq_max, dtype=np.float32
            )  # important to specify dtype here

            traces0 = recording_filtered.get_traces(start_frame=0, end_frame=int(1 * recording_filtered.get_sampling_frequency()))
            traces0 = traces0.astype(dtype=traces0.dtype, order='C')

            # noise_level = estimate_noise_level(traces0)
            # print(f'Noise level: {noise_level}')
            # scale_factor = qfc_estimate_quant_scale_factor(traces0, target_residual_stdev=noise_level * 0.2)

            if compression_type == 'qfc':
                compression_method = compression_opts['compression_method']
                target_compression_ratio = compression_opts['target_compression_ratio']
                zlib_level = compression_opts['zlib_level']
                zstd_level = compression_opts['zstd_level']

                if target_compression_ratio > 0:
                    scale_factor = qfc_estimate_quant_scale_factor(
                        traces0,
                        target_compression_ratio=target_compression_ratio,
                        compression_method=compression_method,
                        zlib_level=zlib_level,
                        zstd_level=zstd_level
                    )
                    codec = QFCCodec(
                        quant_scale_factor=scale_factor,
                        dtype='float32',
                        segment_length=int(recording_filtered.get_sampling_frequency() * 1),
                        compression_method=compression_method,
                        zlib_level=zlib_level,
                        zstd_level=zstd_level
                    )
                else:
                    if compression_method == 'zlib':
                        codec = 'gzip'
                    else:
                        raise ValueError(f'Compression method {compression_method} not recognized')
            else:
                raise ValueError(f'Compression type {compression_type} not recognized')

            # clear the units group if it exists
            if 'units' in f:
                del f['units']

            with pynwb.NWBHDF5IO(file=f, mode='a') as io:
                nwbfile = io.read()

                electrical_series = nwbfile.acquisition[electrical_series_name]  # type: ignore
                electrical_series_pre = ElectricalSeries(
                    name=electrical_series_name + "_pre",
                    data=pynwb.H5DataIO(
                        recording_filtered.get_traces(),  # type: ignore
                        chunks=(int(recording.get_sampling_frequency() * 1), recording.get_num_channels()),
                        compression=codec
                    ),
                    electrodes=electrical_series.electrodes,
                    starting_time=0.0,  # timestamp of the first sample in seconds relative to the session start time
                    rate=recording_filtered.get_sampling_frequency(),
                )
                nwbfile.add_acquisition(electrical_series_pre)  # type: ignore
                io.write(nwbfile)  # type: ignore

        shutil.move(output_fname_tmp, output_fname)


def _run_spike_sorting(*, input_fname: str, electrical_series_name: str, output_fname: str):
    if os.path.exists(output_fname):
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        output_fname_tmp = tmpdir + '/output.nwb.lindi'
        shutil.copyfile(input_fname, output_fname_tmp)
        cache = lindi.LocalCache()
        with lindi.LindiH5pyFile.from_lindi_file(output_fname_tmp, mode="r+", local_cache=cache) as f:
            # Load recording
            recording = NwbRecordingExtractor(
                h5py_file=f, electrical_series_path='/acquisition/' + electrical_series_name
            )
            print("Whitening")
            # see comment in _scale_recording_if_float_type
            recording_scaled = _scale_recording_if_float_type(recording)
            recording_whitened: si.BaseRecording = spre.whiten(
                recording_scaled,
                dtype="float32",
                num_chunks_per_segment=1,  # by default this is 20 which takes a long time to load depending on the chunking
                chunk_size=int(1e5),
            )
            recording_binary = make_float32_recording(
                recording_whitened, dirname=tmpdir + "/float32_recording"
            )
            sorting = ss.run_sorter('mountainsort5', recording=recording_binary, output_folder=tmpdir + '/sorting_mountainsort5')
            assert isinstance(sorting, si.BaseSorting)
            print('Unit IDs:', sorting.get_unit_ids())
            with pynwb.NWBHDF5IO(file=f, mode='a') as io:
                nwbfile = io.read()
                for ii, unit_id in enumerate(sorting.get_unit_ids()):
                    st = sorting.get_unit_spike_train(unit_id) / sorting.get_sampling_frequency()
                    nwbfile.add_unit(id=ii + 1, spike_times=st)  # must be an int # type: ignore
                io.write(nwbfile)  # type: ignore
        shutil.move(output_fname_tmp, output_fname)


if __name__ == '__main__':
    ephys_compression_study()
