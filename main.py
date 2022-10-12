"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Import whatever libraries/modules you need

import argparse
import logging

from bootcamp.runner import Runner

# Your working code here

# Logging setup
logger = logging.getLogger("bootcamp")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s [%(name)s]')
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    """Starts the bootcamp program and parses the arguments."""

    parser = argparse.ArgumentParser()
    cmd_help_str: str = """Command to run. 'train' will train a new neural network and store it in a new file, 'test'
                        will load a neural network from the latest file, 'summary' will print the summary of the
                        network."""

    parser.add_argument("command", help=cmd_help_str, choices=["train", "test", "summary"])

    parser.add_argument("-f", "--file", help="File to load or save the neural network to/from.", type=str)

    args = parser.parse_args()

    logger.debug("Starting bootcamp program")
    runner = Runner(args.command, args.file)
    runner.run()


