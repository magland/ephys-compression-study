from pairio.sdk import App
from EphysPreprocess import EphysPreprocess
from MountainSort5 import MountainSort5

app = App(
    app_name='ephys_compression_study',
    description='Ephys compression study',
)

app.add_processor(EphysPreprocess)
app.add_processor(MountainSort5)

if __name__ == '__main__':
    app.run()
