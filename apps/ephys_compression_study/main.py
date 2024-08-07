from pairio.sdk import App
from EphysPreprocess import EphysPreprocess

app = App(
    app_name='ephys_compression_study',
    description='Ephys compression study',
)

app.add_processor(EphysPreprocess)

if __name__ == '__main__':
    app.run()
