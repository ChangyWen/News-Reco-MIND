from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.io.mind_all_iterator import MINDAllIterator
from recommenders.models.newsrec.models.nrms import NRMSModel
# from recommenders.models.newsrec.models.naml import NAMLModel
from naml.naml import NAMLModel
import sys
import time
from tqdm import tqdm
import numpy as np
import zipfile


if __name__ == '__main__':
    mind_type = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    model_name = sys.argv[4] if len(sys.argv) > 4 else 'nrms'
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

    directory = f'../data/{mind_type}/MIND{mind_type}_'
    yaml_file = directory + f'utils/{model_name}.yaml'
    wordEmb_file = directory + 'utils/embedding.npy'
    wordDict_file = directory + 'utils/word_dict.pkl'
    userDict_file = directory + 'utils/uid2index.pkl'
    vertDict_file = directory + 'utils/vert_dict.pkl'
    subvertDict_file = directory + 'utils/subvert_dict.pkl'

    train_news_file = directory + 'train/news.tsv'
    train_behaviors_file = directory + 'train/behaviors.tsv'
    # valid_news_file = directory + 'dev/news.tsv'
    # valid_behaviors_file = directory + 'dev/behaviors.tsv'
    valid_news_file = f'../data/demo/MINDdemo_dev/news.tsv'
    valid_behaviors_file = f'../data/demo/MINDdemo_dev/behaviors.tsv'

    hparams = prepare_hparams(
        yaml_file=yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        vertDict_file=vertDict_file,
        subvertDict_file=subvertDict_file,
        batch_size=batch_size,
        epochs=epochs,
        body_size=50,
        save_model=True,
        save_epoch=1,
        show_step=1000
    )
    print(hparams)

    model = None
    if model_name == 'nrms':
        model = NRMSModel(hparams, MINDAllIterator, seed=seed)
    elif model_name == 'naml':
        model = NAMLModel(hparams, MINDAllIterator, seed=seed)
    else:
        raise ValueError(f'Model {model_name} not supported')

    model.fit(
        train_news_file=train_news_file,
        train_behaviors_file=train_behaviors_file,
        valid_news_file=valid_news_file,
        valid_behaviors_file=valid_behaviors_file
    )
    model_path = f'../models/weights_{model_name}_{int(time.time())}.h5'
    model.model.save_weights(model_path)
    print(f'Saved model to path {model_path}')

    print('Running fast evaluation for prediction.txt ...')
    group_impr_indexes, group_labels, group_preds = model.run_fast_eval(
        news_filename='../data/large/MINDlarge_test/news.tsv',
        behaviors_file='../data/large/MINDlarge_test/behaviors.tsv'
    )

    with open('./prediction.txt', 'w') as f:
        for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
            impr_index += 1
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
            f.write(' '.join([str(impr_index), pred_rank])+ '\n')

    f = zipfile.ZipFile(f'./prediction.zip', 'w', zipfile.ZIP_DEFLATED)
    f.write('./prediction.txt', arcname='prediction.txt')
    f.close()
