from recommenders.datasets.mind import (
    download_and_extract_glove,
    URL_MIND_DEMO_TRAIN, URL_MIND_DEMO_VALID, URL_MIND_DEMO_UTILS,
    URL_MIND_SMALL_TRAIN, URL_MIND_SMALL_VALID, URL_MIND_SMALL_UTILS,
    URL_MIND_LARGE_TRAIN, URL_MIND_LARGE_VALID, URL_MIND_LARGE_TEST, URL_MIND_LARGE_UTILS,
)
from recommenders.datasets.download_utils import unzip_file, maybe_download

data_paths = {
    'demo': [URL_MIND_DEMO_TRAIN, URL_MIND_DEMO_VALID, URL_MIND_DEMO_UTILS],
    'small': [URL_MIND_SMALL_TRAIN, URL_MIND_SMALL_VALID, URL_MIND_SMALL_UTILS],
    'large': [URL_MIND_LARGE_TRAIN, URL_MIND_LARGE_VALID, URL_MIND_LARGE_UTILS, URL_MIND_LARGE_TEST]
}

if __name__ == "__main__":
    mind_types = ['demo', 'small']
    # download data
    for mind_type in mind_types:
        for path in data_paths[mind_type]:
            print('downloading {}'.format(path))
            downloaded_zip = maybe_download(url=path, work_directory='../data/{}/'.format(mind_type))
            print('downloaded {}'.format(downloaded_zip))
            unzip_file(zip_src=downloaded_zip, dst_dir=downloaded_zip[:-4])
            print('unziped {}'.format(downloaded_zip))



