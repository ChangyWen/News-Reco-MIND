from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.io.mind_all_iterator import MINDAllIterator as MINDAllIterator_train
from naml.iterator import MINDAllIterator as MINDAllIterator_test
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
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    directory = f'../data/{mind_type}/MIND{mind_type}_'
    yaml_file = directory + f'utils/naml.yaml'
    wordEmb_file = directory + 'utils/embedding.npy'
    wordDict_file = directory + 'utils/word_dict.pkl'
    userDict_file = directory + 'utils/uid2index.pkl'
    vertDict_file = directory + 'utils/vert_dict.pkl'
    subvertDict_file = directory + 'utils/subvert_dict.pkl'

    train_news_file = directory + 'train/news.tsv'
    train_behaviors_file = directory + 'train/behaviors.tsv'
    valid_news_file = directory + 'dev/news.tsv'
    valid_behaviors_file = directory + 'dev/behaviors.tsv'
    test_news_file = directory + 'test/news.tsv'
    test_behaviors_file = directory + 'test/behaviors.tsv'

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
        show_step=1000
    )
    print(hparams)

    model = NAMLModel(hparams, MINDAllIterator_train, MINDAllIterator_test, seed=seed)

    model.fit(
        train_news_file=train_news_file,
        train_behaviors_file=train_behaviors_file,
        valid_news_file=valid_news_file,
        valid_behaviors_file=valid_behaviors_file,
        test_news_file=test_news_file,
        test_behaviors_file=test_behaviors_file,
    )

    time_str = int(time.time())
    model_path = f'../models/weights_naml_{time_str}'
    model.save(model_path)
    print(f'Saved model to {model_path}')

    print('Running fast evaluation for prediction.txt ...')
    group_impr_indexes, group_labels, group_preds = model.run_fast_eval(
        news_filename=test_news_file,
        behaviors_file=test_behaviors_file
    )

    with open(f'./prediction_{time_str}.txt', 'w') as f:
        for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
            impr_index += 1
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
            f.write(' '.join([str(impr_index), pred_rank])+ '\n')

    f = zipfile.ZipFile(f'./prediction_{time_str}.zip', 'w', zipfile.ZIP_DEFLATED)
    f.write(f'./prediction_{time_str}.txt', arcname=f'prediction_{time_str}.txt')
    f.close()
