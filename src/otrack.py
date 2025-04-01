class OTrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class OTrack(BaseTrack):
    def __init__(self, box, conf, time):

        # wait activate
        self.box=box
        self.conf=conf
        self.state = OTrackState.New

    def predict(self, dt=1.0):