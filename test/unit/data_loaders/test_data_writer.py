from corerl.data_loaders.data_writer import DataWriter
from datetime import datetime, UTC

def test_writing_datapt():
    data_writer = DataWriter(username='postgres', password='password', db_name='sensors')

    ts = datetime.now(tz=UTC)
    sensor_name = "orp"
    sensor_val = 780.0

    data_writer.write(timestamp=ts, name=sensor_name, val=sensor_val)
