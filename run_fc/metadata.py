import logging
import os

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


class Metadata:
    def __init__(self, output_path):
        super(Metadata, self).__init__()
        self.messages = []
        self.output_path = output_path
        self.fig = None
    def append(self, message):
        logger.info(message)
        self.messages.append(message)

    def addfig(self, fig):
        self.fig = fig

    def close(self):
        if self.output_path is not None:
            info_path = os.path.join(self.output_path, "info.txt")
            f = open(info_path, "w")
            for line in self.messages:
                f.write(line)
                f.write("\n")
            f.close()
        if self.fig is not None:
            img_path = os.path.join(self.output_path, "loss.png")
            self.fig.savefig(img_path)
