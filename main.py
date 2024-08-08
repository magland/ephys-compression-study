from pairio.client import PairioJobDefinition, PairioJobOutputFile, PairioJobParameter, PairioJobInputFile, PairioJobRequiredResources, submit_job


def main():
    # https://neurosift.app/?p=/nwb&url=https://api.dandiarchive.org/api/assets/2e6b590a-a2a4-4455-bb9b-45cc3d7d7cc0/download/&dandisetId=000463&dandisetVersion=draft
    url = "https://api.dandiarchive.org/api/assets/2e6b590a-a2a4-4455-bb9b-45cc3d7d7cc0/download/"
    electrical_series_name = 'ElectricalSeries'

    service_name = 'hello_world_service'

    compression_ratios = [0, 4, 8, 12, 16, 20, 25, 30, 40, 50, 75, 100]

    job_def1 = PairioJobDefinition(
        appName='ephys_compression_study',
        processorName='ephys_compression_study.ephys_preprocess',
        inputFiles=[
            PairioJobInputFile(
                name='input',
                fileBaseName='input.nwb',
                url=url
            )
        ],
        outputFiles=[
            PairioJobOutputFile(
                name='output',
                fileBaseName='output.nwb.lindi'
            )
        ],
        parameters=[
            PairioJobParameter(
                name='electrical_series_path',
                value='/acquisition/' + electrical_series_name
            ),
            PairioJobParameter(
                name='compression_ratios',
                value=compression_ratios
                # value=[0]
            ),
            PairioJobParameter(
                name='freq_min',
                value=300
            ),
            PairioJobParameter(
                name='freq_max',
                value=6000
            )
        ]
    )
    required_resources = PairioJobRequiredResources(
        numCpus=4,
        numGpus=0,
        memoryGb=4,
        timeSec=60 * 60 * 4
    )
    job1 = submit_job(
        service_name=service_name,
        job_definition=job_def1,
        required_resources=required_resources,
        tags=['ephys-compression-study', '000463'],
        rerun_failing=True
    )
    print(job1.job_url, job1.status)

    job_def2 = PairioJobDefinition(
        appName='ephys_compression_study',
        processorName='ephys_compression_study.mountainsort5',
        inputFiles=[
            PairioJobInputFile(
                name='input',
                fileBaseName='input.nwb.lindi',
                url=job1.get_output('output')
            )
        ],
        outputFiles=[
            PairioJobOutputFile(
                name='output',
                fileBaseName='output.nwb.lindi'
            )
        ],
        parameters=[
            PairioJobParameter(
                name='electrical_series_path',
                value='/acquisition/' + electrical_series_name
            ),
            PairioJobParameter(
                name='compression_ratios',
                value=compression_ratios
                # value=[0]
            )
        ]
    )
    required_resources = PairioJobRequiredResources(
        numCpus=4,
        numGpus=0,
        memoryGb=4,
        timeSec=60 * 60 * 4
    )
    job2 = submit_job(
        service_name=service_name,
        job_definition=job_def2,
        required_resources=required_resources,
        tags=['ephys-compression-study', '000463'],
        rerun_failing=True
    )
    print(job2.job_url, job2.status)


if __name__ == '__main__':
    main()
