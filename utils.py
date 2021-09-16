from tensorbay import GAS
from tensorbay.dataset import Segment
from  gas_key import KEY

def get_data_from_gas(name = "spherical_segmentation", segement = "2d3ds_sphere"):
    gas = GAS(KEY)
    dataset_client = gas.get_dataset(name)
    segments = dataset_client.list_segment_names()
    return Segment(segement, dataset_client)

