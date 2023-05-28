from utils import Utils
from sound_parser import SoundParser
from constants import sample_path


def main():
    parser = SoundParser(sample_path)
    Utils.show_graphs()


if __name__ == "__main__":
    main()
