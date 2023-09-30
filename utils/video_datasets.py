import pandas as pd
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """ Video Dataset for loading video.
        It will output only path of video (neither video file path or video folder path).
        However, you can load video as torch.Tensor (C x L x H x W).
        See below for an example of how to read video as torch.Tensor.
        Your video dataset can be image frames or video files.

    Args:
        csv_file (str): path fo csv file which store path of video file or video folder.
            the format of csv_file should like:

            # example_video_file.csv   (if the videos of dataset is saved as video file)

            path
            ~/path/to/video/file1.mp4
            ~/path/to/video/file2.mp4
            ~/path/to/video/file3.mp4
            ~/path/to/video/file4.mp4

            # example_video_folder.csv   (if the videos of dataset is saved as image frames)

            path
            ~/path/to/video/folder1/
            ~/path/to/video/folder2/
            ~/path/to/video/folder3/
            ~/path/to/video/folder4/

    Example:

        if the videos of dataset is saved as video file

        if the video of dataset is saved as frames in video folder
        The tree like: (The names of the images are arranged in ascending order of frames)
        ~/path/to/video/folder1
        ├── frame-001.jpg
        ├── frame-002.jpg
        ├── frame-003.jpg
        └── frame-004.jpg

    """

    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video """
        video = self.dataframe.iloc[index].path
        if self.transform:
            video = self.transform(video)
        return video
