from humansegment_unet import Segment

def create_segment():
    return Segment(max_size=512, outline_thickness=1)
