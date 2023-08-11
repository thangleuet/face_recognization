import queue, threading

# bufferless VideoCapture
class CameraCapture:
    def __init__(self, capdevice):
        self.cap = capdevice
        self.q = queue.Queue(1)
        self.no_stop_request = True
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.no_stop_request:
            ret, frame = self.cap.read()

            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous(unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        frame = self.q.get()
        if frame is not None:
            ret = True
        else:
            ret = False
        return ret, frame

    def close(self, timeout_sec):
        self.no_stop_request = False
        if self.t.is_alive():
            self.t.join(timeout_sec)