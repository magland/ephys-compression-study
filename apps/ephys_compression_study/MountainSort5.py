from typing import List
import os
import shutil
import numpy as np
from pairio.sdk import ProcessorBase, BaseModel, Field, InputFile, OutputFile

class MountainSort5Context(BaseModel):
    input: InputFile = Field(description='Input NWB file in .nwb or .nwb.lindi format')
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
        import spikeinterface.sorters as ss
        import spikeinterface as si

        QFCCodec.register_codec()

        cache = lindi.LocalCache(cache_dir='lindi_cache')

        if context.input.file_base_name.endswith('.nwb'):
            print('Creating LINDI file from NWB file')
            url = context.input.get_url()
            assert url, 'No URL for input file'
            with lindi.LindiH5pyFile.from_hdf5_file(url) as f:
                f.write_lindi_file('output.nwb.lindi')
        elif context.input.file_base_name.endswith('.lindi.json') or context.input.file_base_name.endswith('.lindi'):
            print('Creating LINDI file')
            url = context.input.get_url()
            assert url, 'No URL for input file'
            with lindi.LindiH5pyFile.from_lindi_file(url) as f:
                f.write_lindi_file('output.nwb.lindi')
        else:
            raise Exception(f'Unexpected file extension: {context.input.file_base_name}')

        with lindi.LindiH5pyFile.from_lindi_file('output.nwb.lindi', mode="r+", local_cache=cache) as f:
            with pynwb.NWBHDF5IO(file=f, mode='a') as io:
                nwbfile = io.read()
                for compression_ratio in context.compression_ratios:
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
                        shutil.rmtree('recording_float32')
                    recording_binary = make_float32_recording(recording_whitened, dirname='recording_float32')
                    if os.path.exists('sorting_mountainsort5'):
                        shutil.rmtree('sorting_mountainsort5')
                    sorting = ss.run_sorter('mountainsort5', recording=recording_binary, output_folder='sorting_mountainsort5')
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
                        spike_times_index.append(ii)
                        ii += len(st)
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
