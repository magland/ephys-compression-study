from typing import List
import os
import shutil
import numpy as np
from pairio.sdk import ProcessorBase, BaseModel, Field, InputFile, OutputFile

class MountainSort5Context(BaseModel):
    input: InputFile = Field(description='Input NWB file in .nwb.lindi format')
    output: OutputFile = Field(description='Output data in .nwb.lindi format')
    electrical_series_path: str = Field(description='Path to the electrical series object in the NWB file')
    compression_ratios: List[float] = Field(description='List of compression ratios to use')


class MountainSort5(ProcessorBase):
    name = 'ephys_compression_study.mountainsort5'
    description = 'Run mountainsort5 on an electrophysiology dataset prepared with ephys_compression_study.ephys_preprocess with a series of compression ratios'
    label = 'ephys_compression_study.mountainsort5'
    image = 'magland/pairio-ephys-compression-study:0.1.0'
    executable = '/app/main.py'
    attributes = {}

    @staticmethod
    def run(
        context: MountainSort5Context
    ):
        import uuid
        import pynwb
        import lindi
        from qfc.codecs.QFCCodec import QFCCodec
        from helpers.nwbextractors import NwbRecordingExtractor
        from helpers.make_float32_recording import make_float32_recording
        from helpers._scale_recording_if_float_type import _scale_recording_if_float_type
        import spikeinterface.preprocessing as spre
        import spikeinterface as si
        import mountainsort5 as ms5

        QFCCodec.register_codec()

        cache = lindi.LocalCache(cache_dir='lindi_cache')

        if not context.input.file_base_name.endswith('.nwb.lindi'):
            raise Exception(f'Unexpected file extension: {context.input.file_base_name}')
        context.input.download('output.nwb.lindi')

        with lindi.LindiH5pyFile.from_lindi_file('output.nwb.lindi', mode="r+", local_cache=cache) as f:
            with pynwb.NWBHDF5IO(file=f, mode='a') as io:
                nwbfile = io.read()
                for compression_ratio in context.compression_ratios:
                    print('\n\n\n\n\n\n')
                    print(f'COMPRESSION RATIO: {compression_ratio}')
                    ep = context.electrical_series_path + f"_pre_{compression_ratio}"
                    recording = NwbRecordingExtractor(
                        h5py_file=f, electrical_series_path=ep
                    )
                    recording_scaled = _scale_recording_if_float_type(recording)
                    recording_whitened: si.BaseRecording = spre.whiten(
                        recording_scaled,
                        dtype="float32",
                        num_chunks_per_segment=1,  # by default this is 20 which takes a long time to load depending on the chunking
                        chunk_size=int(1e5),
                    )
                    if os.path.exists('recording_float32'):
                        shutil.rmtree('recording_float32', ignore_errors=True)
                    recording_binary = make_float32_recording(recording_whitened, dirname='recording_float32')

                    print("Setting up sorting parameters")
                    detect_threshold = 5
                    scheme1_detect_channel_radius = None
                    detect_time_radius_msec = 0.5
                    detect_sign = -1
                    snippet_T1 = 20
                    snippet_T2 = 50
                    snippet_mask_radius = None
                    npca_per_channel = 3
                    npca_per_subdivision = 10
                    scheme1_sorting_parameters = ms5.Scheme1SortingParameters(
                        detect_threshold=detect_threshold,
                        detect_channel_radius=scheme1_detect_channel_radius,
                        detect_time_radius_msec=detect_time_radius_msec,
                        detect_sign=detect_sign,
                        snippet_T1=snippet_T1,
                        snippet_T2=snippet_T2,
                        snippet_mask_radius=snippet_mask_radius,
                        npca_per_channel=npca_per_channel,
                        npca_per_subdivision=npca_per_subdivision,
                    )

                    print("Sorting scheme 1")
                    sorting = ms5.sorting_scheme1(
                        recording=recording_binary,
                        sorting_parameters=scheme1_sorting_parameters,
                    )
                    assert isinstance(sorting, si.BaseSorting)
                    print('Unit IDs:', sorting.get_unit_ids())

                    # in future, use pynwb to write units, but I don't know how to do this yet
                    g_units = f.create_group(f'processing/ecephys/units_mountainsort5_{compression_ratio}')
                    g_units.attrs['colnames'] = ['spike_times']
                    g_units.attrs['description'] = 'Units from mountainsort5 with compression ratio ' + str(compression_ratio)
                    g_units.attrs['namespace'] = 'core'
                    g_units.attrs['neurodata_type'] = 'Units'
                    g_units.attrs['object_id'] = str(uuid.uuid4())
                    ds_id = g_units.create_dataset('id', data=np.arange(len(sorting.get_unit_ids())) + 1)
                    ds_id.attrs['namespace'] = 'hdmf-common'
                    ds_id.attrs['neurodata_type'] = 'ElementIdentifiers'
                    ds_id.attrs['object_id'] = str(uuid.uuid4())
                    spike_times_list = []
                    spike_times_index = []
                    ii = 0
                    for unit_id in sorting.get_unit_ids():
                        st = sorting.get_unit_spike_train(unit_id) / sorting.get_sampling_frequency()
                        spike_times_list.append(st)
                        ii += len(st)
                        spike_times_index.append(ii)
                    spike_times = np.concatenate(spike_times_list)
                    spike_times_index = np.array(spike_times_index)
                    ds_spike_times = g_units.create_dataset('spike_times', data=spike_times)
                    ds_spike_times.attrs['description'] = 'Spike times for each unit'
                    ds_spike_times.attrs['namespace'] = 'hdmf-common'
                    ds_spike_times.attrs['neurodata_type'] = 'VectorData'
                    ds_spike_times.attrs['object_id'] = str(uuid.uuid4())
                    ds_spike_times_index = g_units.create_dataset('spike_times_index', data=spike_times_index)
                    ds_spike_times_index.attrs['description'] = 'Index of spike times for each unit'
                    ds_spike_times_index.attrs['namespace'] = 'hdmf-common'
                    ds_spike_times_index.attrs['neurodata_type'] = 'VectorIndex'
                    ds_spike_times_index.attrs['object_id'] = str(uuid.uuid4())
                    ds_spike_times_index.attrs['target'] = ds_spike_times.ref

                io.write(nwbfile)  # type: ignore

        print('Uploading output file')
        context.output.upload('output.nwb.lindi')

def estimate_noise_level(traces):
    noise_level = np.median(np.abs(traces - np.median(traces))) / 0.6745
    return noise_level
