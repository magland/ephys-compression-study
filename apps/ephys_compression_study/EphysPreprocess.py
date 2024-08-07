from typing import List
from pairio.sdk import ProcessorBase, BaseModel, Field, InputFile, OutputFile

class EphysPreprocessContext(BaseModel):
    input: InputFile = Field(description='Input NWB file in .nwb or .nwb.lindi format')
    output: OutputFile = Field(description='Output data in .nwb.lindi format')
    electrical_series_path: str = Field(description='Path to the electrical series object in the NWB file')
    compression_ratios: List[float] = Field(description='List of compression ratios to use')
    freq_min: float = Field(default=300, description='Minimum frequency for bandpass filter')
    freq_max: float = Field(default=6000, description='Maximum frequency for bandpass filter')


class EphysPreprocess(ProcessorBase):
    name = 'ephys_compression_study.ephys_preprocess'
    description = 'Run preprocessing on an electrophysiology dataset with a series of compression ratios'
    label = 'ephys_compression_study.ephys_preprocess'
    image = 'magland/pairio-ephys-compression-study:0.1.0'
    executable = '/app/main.py'
    attributes = {}

    @staticmethod
    def run(
        context: EphysPreprocessContext
    ):
        import numpy as np
        import pynwb
        from pynwb.ecephys import ElectricalSeries
        import lindi
        from qfc import qfc_estimate_quant_scale_factor
        from qfc.codecs.QFCCodec import QFCCodec
        from helpers.nwbextractors import NwbRecordingExtractor
        import spikeinterface.preprocessing as spre

        QFCCodec.register_codec()

        cache = lindi.LocalCache()

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
            recording = NwbRecordingExtractor(
                h5py_file=f, electrical_series_path=context.electrical_series_path
            )

            # bandpass filter
            freq_min = context.freq_min
            freq_max = context.freq_max
            recording_filtered = spre.bandpass_filter(
                recording, freq_min=freq_min, freq_max=freq_max, dtype=np.float32
            )  # important to specify dtype here

            traces0 = recording_filtered.get_traces(start_frame=0, end_frame=int(1 * recording_filtered.get_sampling_frequency()))
            traces0 = traces0.astype(dtype=traces0.dtype, order='C')

            # noise_level = estimate_noise_level(traces0)
            # print(f'Noise level: {noise_level}')
            # scale_factor = qfc_estimate_quant_scale_factor(traces0, target_residual_stdev=noise_level * 0.2)
            for compression_ratio in context.compression_ratios:
                print(f'Compression ratio: {compression_ratio}')
                compression_method = 'zlib'
                target_compression_ratio = compression_ratio
                zlib_level = 3
                zstd_level = 3

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

                with pynwb.NWBHDF5IO(file=f, mode='a') as io:
                    nwbfile = io.read()

                    electrical_series = nwbfile.acquisition[context.electrical_series_path]  # type: ignore
                    electrical_series_name = context.electrical_series_path.split('/')[-1]
                    electrical_series_pre = ElectricalSeries(
                        name=electrical_series_name + f"_pre_{compression_ratio}",
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
